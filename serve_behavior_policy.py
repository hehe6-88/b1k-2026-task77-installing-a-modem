#!/usr/bin/env python3
# =============================================================================
# serve_behavior_policy.py -- Serve our trained pi0.5 policy to OmniGibson's
# evaluator over BEHAVIOR's websocket protocol.
#
#   python scripts/serve_behavior_policy.py \
#       --checkpoint /workspace/checkpoints/ours/behavior_boxing_books/boxing_books_lora_v1/29999
#
#   # then, in the conda `behavior` env:
#   python -m omnigibson.eval.eval --task-name boxing_books_up_for_storage \
#       --policy websocket --host 127.0.0.1 --port 8000 ...
#
# ---------------------------------------------------------------------------
# WHY A SEPARATE PROCESS
#
# The policy needs openpi (JAX); the simulator needs OmniGibson (Isaac Sim).
# Those two dependency sets genuinely conflict -- OmniGibson forces
# websockets>=15 while isaacsim-kernel pins ==12.0, among others. Running them
# as two processes over a socket is not a workaround, it is the architecture the
# challenge itself specifies for submissions ("a Docker image serving the policy
# via WebSocket").
#
# So this file speaks BEHAVIOR's wire protocol directly rather than importing
# omnigibson.eval.utils.network_utils, which would drag the simulator's
# dependencies into openpi's venv.
#
# ---------------------------------------------------------------------------
# THE PROTOCOL (read off omnigibson/eval/utils/network_utils.py, 2026-08-12)
#
#   server -> client, on connect : msgpack(metadata dict)
#   client -> server             : msgpack(obs)  or  msgpack({"reset": True})
#   server -> client             : msgpack({"action": np.ndarray, ...})
#
# A missing "action" key makes the client retry twice, then raise. Arrays use
# the msgpack-numpy {b"__ndarray__": ...} encoding reproduced below.
#
# ---------------------------------------------------------------------------
# ONE ACTION PER QUERY
#
# The evaluator asks for a SINGLE action vector per call and re-uses it while
# obs["need_new_action"] is false. Our model emits ACTION_HORIZON (30) actions
# per inference, so openpi's ActionChunkBroker holds the chunk and hands out one
# action at a time, re-inferring only when it runs out.
# =============================================================================

from __future__ import annotations

import argparse
import asyncio
import functools
import http
import logging
import os
import pathlib
import sys
import traceback

import msgpack
import numpy as np
import websockets.asyncio.server

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("behavior-policy-server")

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

# Safe to import at module level: task_prompt and warm_start pull in only the
# standard library, unlike configs/ and openpi which drag in JAX and are
# imported inside main().
from src import task_prompt, warm_start  # noqa: E402


# --- msgpack, matching BEHAVIOR's encoding ------------------------------------
# Adapted from the same msgpack-numpy scheme both openpi and BEHAVIOR use. We
# only ever RECEIVE arrays (the client converts its torch tensors to ndarray
# before packing) and only ever SEND one, so this is deliberately minimal.


def _pack(obj):
    if isinstance(obj, np.ndarray):
        if obj.dtype.kind in ("V", "O", "c"):
            raise ValueError(f"unsupported dtype: {obj.dtype}")
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


def _unpack(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


Packer = functools.partial(msgpack.Packer, default=_pack)
unpackb = functools.partial(msgpack.unpackb, object_hook=_unpack)


# --- observation adapter ------------------------------------------------------
# THE most failure-prone part of this file, and the reason scripts/dump_eval_obs.py
# exists. BehaviorInputs was written against the DATASET's field names; the
# simulator uses its own. Mapping them wrong does not necessarily raise -- it can
# feed the model plausible-looking garbage -- so every lookup here fails loudly
# with the keys that were actually present.

# The prompt is NOT hardcoded here any more -- it is read from the converted
# dataset's meta/tasks.jsonl at startup, which is the same file training reads
# when prompt_from_task=True. See src/task_prompt.py for why.
#
# A hardcoded copy here is exactly the bug that cost us a 30,000-step run: it
# can drift from the dataset with no error, no warning, and no log line, and
# the only symptom is a policy that scores badly for reasons that look like bad
# hyperparameters.
#
# ⚠️ To serve the OLD v1 checkpoint (trained on the snake_case task name), pass
# the old string explicitly:
#      --prompt boxing_books_up_for_storage

DATASET_BASE = "observation.rgb.zed_link_camera_0"
DATASET_LEFT = "observation.rgb.left_realsense_link_camera_0"
DATASET_RIGHT = "observation.rgb.right_realsense_link_camera_0"
BEHAVIOR_STATE_DIM = 61

# Substrings that identify each camera in the simulator's key names. The
# evaluator resolves camera names via get_robot_camera_names, so the exact
# strings depend on the robot config; matching on these stems is more robust
# than hardcoding a full name.
CAMERA_HINTS = {
    DATASET_BASE: ("zed", "head"),
    DATASET_LEFT: ("left_realsense", "left_wrist", "left"),
    DATASET_RIGHT: ("right_realsense", "right_wrist", "right"),
}


def _find_camera_key(obs: dict, hints: tuple[str, ...]) -> str:
    rgb_keys = [k for k in obs if k.endswith("::rgb")]
    for hint in hints:
        for key in rgb_keys:
            if hint in key.lower():
                return key
    raise KeyError(
        f"no RGB key matching {hints}. RGB keys present: {rgb_keys}. "
        "Re-run scripts/dump_eval_obs.py -- the robot config's camera names may have changed."
    )


def _find_state(obs: dict) -> np.ndarray:
    """Locate the 61-dim proprioception vector.

    Checked by WIDTH rather than by name: BehaviorInputs validates the state is
    exactly 61 wide, and a silently wrong vector of the right width is the worst
    outcome, so we prefer an explicit named match and fall back to width only as
    a last resort -- reporting what we chose either way.
    """
    for name in ("proprio", "observation.state", "state"):
        for key, value in obs.items():
            if name in str(key).lower() and getattr(value, "shape", None):
                arr = np.asarray(value, dtype=np.float32).reshape(-1)
                if arr.shape[-1] == BEHAVIOR_STATE_DIM:
                    return arr
    candidates = {
        str(k): list(v.shape)
        for k, v in obs.items()
        if getattr(v, "shape", None) and np.asarray(v).reshape(-1).shape[-1] == BEHAVIOR_STATE_DIM
    }
    if len(candidates) == 1:
        key = next(iter(candidates))
        logger.warning("proprioception matched by width only, using %r", key)
        return np.asarray(obs[key], dtype=np.float32).reshape(-1)
    raise KeyError(
        f"could not find a {BEHAVIOR_STATE_DIM}-dim state. "
        f"{BEHAVIOR_STATE_DIM}-wide candidates: {candidates}. All keys: {sorted(str(k) for k in obs)}"
    )


def _rgb(obs: dict, hints: tuple[str, ...]) -> np.ndarray:
    """Fetch a camera image as 3-channel RGB.

    The SIMULATOR returns RGBA -- (224, 224, 4) uint8, confirmed against a live
    environment on 2026-08-12 -- while the DATASET stores RGB. pi0.5 was trained
    on 3 channels, so the alpha channel must be dropped here; passing 4 channels
    through would at best raise deep inside the vision tower and at worst be
    reinterpreted as a differently-shaped image.
    """
    arr = np.asarray(obs[_find_camera_key(obs, hints)])
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected an (H, W, 3) image after dropping alpha, got {arr.shape}")
    return arr


def adapt_observation(obs: dict, prompt: str) -> dict:
    """Simulator observation -> the dict BehaviorInputs expects."""
    return {
        DATASET_BASE: _rgb(obs, CAMERA_HINTS[DATASET_BASE]),
        DATASET_LEFT: _rgb(obs, CAMERA_HINTS[DATASET_LEFT]),
        DATASET_RIGHT: _rgb(obs, CAMERA_HINTS[DATASET_RIGHT]),
        "observation.state": _find_state(obs),
        "prompt": prompt,
    }


# --- server -------------------------------------------------------------------


class BehaviorPolicyServer:
    def __init__(self, policy, prompt: str, host: str, port: int):
        self._policy = policy
        self._prompt = prompt
        self._host = host
        self._port = port
        self._logged_keys = False

    async def _handler(self, websocket):
        logger.info("client connected: %s", websocket.remote_address)
        packer = Packer()
        await websocket.send(packer.pack({"server": "behavior-pi05", "prompt": self._prompt}))

        while True:
            try:
                request = unpackb(await websocket.recv(), strict_map_key=False)

                if "reset" in request:
                    # Drop the half-consumed action chunk; the next query starts
                    # a fresh episode and must not replay the previous one.
                    if hasattr(self._policy, "reset"):
                        self._policy.reset()
                    logger.info("policy reset")
                    continue

                if not self._logged_keys:
                    logger.info("first observation keys: %s", sorted(str(k) for k in request))
                    self._logged_keys = True

                result = self._policy.infer(adapt_observation(request, self._prompt))
                action = np.asarray(result["actions"], dtype=np.float32)
                await websocket.send(packer.pack({"action": action}))

            except websockets.exceptions.ConnectionClosed:
                logger.info("client disconnected")
                break
            except Exception:
                # Send the traceback as a string: the client raises RuntimeError
                # on a str reply, which surfaces the real error at the evaluator
                # instead of a mystery timeout.
                logger.error("inference failed:\n%s", traceback.format_exc())
                await websocket.send(traceback.format_exc())

    @staticmethod
    def _health_check(connection, request):
        """Answer the client's HTTP health probe.

        WebsocketClientPolicy._wait_for_server GETs http://host:port/healthz and
        BLOCKS until it returns 200 -- before it ever tries the websocket. A
        server that only speaks websocket therefore never gets a connection, and
        the evaluator sits in a "Health check failed, waiting for server" loop
        forever rather than failing. Found the hard way on 2026-08-12.
        """
        if getattr(request, "path", None) == "/healthz":
            return connection.respond(http.HTTPStatus.OK, "OK\n")
        return None

    async def _run(self):
        async with websockets.asyncio.server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=self._health_check,
        ):
            logger.info("serving on ws://%s:%d (health check at /healthz)", self._host, self._port)
            await asyncio.Future()

    def serve_forever(self):
        asyncio.run(self._run())


def main() -> None:
    ap = argparse.ArgumentParser(description="Serve the BEHAVIOR pi0.5 policy over websocket.")
    ap.add_argument("--checkpoint", required=True, help="checkpoint step directory (e.g. .../29999)")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--prompt", default=None, help="override the language prompt")
    # Each task has its own config module; the prompt and repo_id are resolved
    # from whichever one is selected. Defaults to task 23.
    ap.add_argument("--config", default="task23_boxing_books",
                    help="task folder under configs/ (default: task23_boxing_books)")
    # ⚠️ REQUIRED WHEN SERVING A MULTI-TASK CHECKPOINT.
    #
    # The cluster {77, 89, 92} is ONE policy whose three behaviours are held
    # apart by language alone. Its dataset holds three prompts, so "the prompt"
    # is not a thing the server can work out by itself -- and the wrong choice
    # does not look like an error. It looks like the robot confidently doing the
    # modem task during a fax-machine rollout, which reads as a policy quality
    # problem in the video and in the q_score.
    ap.add_argument("--task", type=int, default=None,
                    help="upstream challenge task index (e.g. 89) to serve. Required "
                         "for a multi-task checkpoint; ignored for single-task ones.")
    # Pins the policy's sampling RNG so a checkpoint sweep varies only the
    # checkpoint. See the long note at the construction site for exactly what
    # this does and does not buy -- in short: openpi already defaults to a fixed
    # key(0), so this makes the seed EXPLICIT and logged rather than making an
    # otherwise-random policy deterministic, and it does NOT make a rollout
    # reproducible because the simulator is the dominant source of variance.
    ap.add_argument("--seed", type=int, default=None,
                    help="explicitly pin the policy's sampling RNG. Pass the same "
                         "value to every checkpoint in a sweep. Note this does NOT "
                         "make rollouts deterministic -- the simulator dominates.")
    args = ap.parse_args()

    ckpt = pathlib.Path(args.checkpoint)
    if not ckpt.exists():
        sys.exit(f"ERROR: {ckpt} does not exist")

    import importlib

    from openpi.policies import policy_config as _policy_config
    from openpi_client.action_chunk_broker import ActionChunkBroker

    behavior_config = importlib.import_module(f"configs.{args.config}")
    cfg = behavior_config.get_config()

    # A config declares either a single TASK_INDEX or, for the cluster, a
    # TASK_INDICES collection. Check the plural FIRST: a multi-task config may
    # keep a TASK_INDEX around for a demos path or a log line, and preferring
    # the singular would serve three tasks under one task's prompt.
    declared = getattr(behavior_config, "TASK_INDICES", None)
    if declared is None:
        declared = getattr(behavior_config, "TASK_INDEX", None)
    task_indices = warm_start.normalize_task_indices(declared) if declared is not None else ()
    logger.info("config: %s (task %s)", args.config, list(task_indices) or "?")

    serving_task = args.task
    if len(task_indices) > 1:
        if serving_task is None and args.prompt is None:
            sys.exit(
                f"ERROR: {args.config} is a MULTI-TASK config covering "
                f"{list(task_indices)}.\n"
                "       Pass --task <upstream index> to say which one this server is\n"
                "       for, or --prompt to override the language input outright.\n"
                "       There is no safe default: the tasks are held apart by the\n"
                "       prompt, so guessing produces a confident rollout of the\n"
                "       wrong task and nothing in the metrics would say so."
            )
        if serving_task is not None and serving_task not in task_indices:
            sys.exit(
                f"ERROR: --task {serving_task} is not in {args.config}'s task set "
                f"{list(task_indices)}."
            )
    elif serving_task is None and len(task_indices) == 1:
        (serving_task,) = task_indices

    # Resolve the prompt from the dataset that training actually reads, rather
    # than from a constant in this file. See src/task_prompt.py.
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    dataset_root = pathlib.Path(lerobot_home) / cfg.data.repo_id if lerobot_home else None
    try:
        prompt, source = task_prompt.resolve_prompt(
            override=args.prompt, dataset_root=dataset_root, task_index=serving_task
        )
    except ValueError as exc:
        # resolve_prompt raises rather than guessing when a multi-task dataset
        # cannot be narrowed to one task. That is fatal here by design.
        sys.exit(f"ERROR: {exc}")
    task_prompt.warn_if_unverified(source, logger.warning)

    # Even with an explicit --prompt, say out loud when it disagrees with the
    # dataset. Serving v1 deliberately uses the old string, so this is a
    # warning and not an error -- but it must never be silent.
    if dataset_root is not None:
        if serving_task is not None:
            dataset_prompt = task_prompt.prompt_for_source_task(dataset_root, serving_task)
            # Pre-cluster datasets do not record their upstream index, so fall
            # back to the single-task read rather than reporting a false mismatch.
            if dataset_prompt is None:
                dataset_prompt = task_prompt.read_converted_dataset_prompt(dataset_root)
        else:
            dataset_prompt = task_prompt.read_converted_dataset_prompt(dataset_root)
        if dataset_prompt is not None and dataset_prompt != prompt:
            logger.warning("=" * 72)
            logger.warning("PROMPT MISMATCH -- the policy will run out of distribution")
            logger.warning("  serving with : %r", prompt)
            logger.warning("  dataset says : %r", dataset_prompt)
            logger.warning("  (%s/meta/tasks.jsonl)", dataset_root)
            logger.warning("Expected ONLY when serving the old v1 checkpoint.")
            logger.warning("=" * 72)

    logger.info("serving prompt: %r", prompt)
    logger.info("  prompt source: %s", source)
    logger.info("loading policy from %s", ckpt)

    policy = _policy_config.create_trained_policy(cfg, ckpt, default_prompt=prompt)

    # SEEDING: the factory takes no seed, but the Policy it returns does.
    #
    # Checked against upstream openpi 2026-10-09:
    #   policy_config.create_trained_policy(...)  -- NO rng/seed/key parameter,
    #                                                and no **kwargs forwarding
    #   Policy.__init__(..., *, rng: at.KeyArrayLike | None = None, ...)
    #       self._rng = rng or jax.random.key(0)          # policy.py
    #       self._rng, sample_rng = jax.random.split(self._rng)   # per infer()
    #
    # Two consequences that correct our older notes:
    #
    # 1. The default is ALREADY a fixed key(0). So an unseeded server is not
    #    "random" across restarts. But the state ADVANCES per call, so within
    #    one process the second rollout starts from a different state than the
    #    first -- which is the dominant effect when one server serves five
    #    instances in a row, as the sweep does.
    #
    #    ⚠️ Consequence worth fixing later: to compare checkpoints per instance
    #    on equal footing the RNG should be reset at the START OF EACH INSTANCE,
    #    not once per server. Not implemented -- would need a reset hook on the
    #    websocket "reset" message.
    #
    # 2. ⚠️ What actually makes rollouts differ run to run is THE SIMULATOR:
    #    Isaac Sim physics and GPU float nondeterminism, amplified by the fact
    #    that our own actions feed back into the next observation. Seeding the
    #    policy does NOT make a rollout deterministic. "Do not compare single
    #    rollouts" still stands; only its reason changes.
    #
    # Setting a private attribute is not ideal, but it is the only route short
    # of reimplementing the factory -- and it is verified rather than assumed,
    # so a future openpi that renames it fails loudly instead of silently
    # leaving the sweep unseeded while claiming otherwise.
    seeded = False
    effective_seed = None
    if args.seed is not None:
        import jax

        if hasattr(policy, "_rng"):
            policy._rng = jax.random.key(args.seed)
            seeded = True
            effective_seed = args.seed
            logger.info("policy._rng set to jax.random.key(%d)", args.seed)
        else:
            logger.warning("=" * 72)
            logger.warning("--seed %d requested, but the policy object has no _rng", args.seed)
            logger.warning("attribute, so it could NOT be seeded. openpi may have")
            logger.warning("renamed it. Do NOT treat this run as seeded.")
            logger.warning("=" * 72)
    else:
        # Not an error: openpi's own default is a fixed key(0), so an unseeded
        # server is still reproducible -- just not distinguishable in the log
        # from one we seeded deliberately.
        effective_seed = 0
        logger.info("no --seed given; openpi's default fixed key(0) applies")

    # The EFFECTIVE seed, not the requested one -- a sweep's logs must show what
    # actually took effect.
    logger.info("policy seeded explicitly: %s (effective seed: %s)", seeded, effective_seed)
    logger.warning(
        "⚠️  policy seeding does NOT make rollouts deterministic -- Isaac Sim "
        "physics and GPU nondeterminism dominate."
    )

    # One action per query, re-inferring every ACTION_HORIZON calls.
    horizon = cfg.model.action_horizon
    logger.info("wrapping in ActionChunkBroker(action_horizon=%d)", horizon)
    policy = ActionChunkBroker(policy, action_horizon=horizon)

    BehaviorPolicyServer(policy, prompt, args.host, args.port).serve_forever()


if __name__ == "__main__":
    main()

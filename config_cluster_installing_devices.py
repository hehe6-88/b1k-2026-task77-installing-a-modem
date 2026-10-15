# =============================================================================
# cluster_installing_devices/config.py -- ONE policy over the Day-17 cluster.
#
#   77  installing_a_modem        "Install the modem by placing it on the
#                                  cabinet and turning it on."
#   89  installing_a_fax_machine  "Install the fax machine by placing it on the
#                                  cubicle desk and turning it on."
#   92  installing_a_scanner      "Install the scanner by turning it on and
#                                  leaving it on the desk near the laptop."
#
# (Prompts are shown for orientation only. They are NOT hardcoded anywhere --
#  they are read from the converted dataset's meta/tasks.jsonl at runtime. See
#  src/task_prompt.py, and the 30,000-step run that paid for that rule.)
#
# ---------------------------------------------------------------------------
# WHAT THIS RUN IS FOR
#
# Every task in the cluster shares one skill: transport a device to a named
# surface, then actuate its switch. v1's diagnosis was that the policy knows
# WHERE things are and HOW to open and close its hand, but not WHAT to grasp.
# Three tasks that each require grasping the named device rather than the
# nearest object is much better pressure on that weakness than more demos of a
# single task would be.
#
# ⚠️ THE WHOLE THING RESTS ON THE PROMPT REACHING BEHAVIOUR. pi0.5 holds these
# three behaviours apart by language and nothing else. If the language pathway
# is dead, multi-task is WORSE than single-task -- three tasks collapse toward
# an average policy. The prompt diagnostic folded into the start of the pilot
# pod is what reads that, and it is sharpest on THIS checkpoint after training,
# not on checkpoint_2 (which has never seen 77/89/92).
#
# ---------------------------------------------------------------------------
# ⚠️ TASK 0 IS NOT HERE, AND THAT IS NOT AN OVERSIGHT
#
# turning_on_radio was a cluster member on Day 17 and was removed 2026-09-14:
# the radio is already on the table, so it is actuation-only with no transport
# phase, and training on a member that shares half the skill dilutes the half we
# are trying to teach.
#
# It is out of TRAINING, not out of the project -- it remains the prompt
# diagnostic's in-distribution control. Do not "finish the job" by deleting it
# from the diagnostic; that re-opens a question Day 17 already closed.
#
# ---------------------------------------------------------------------------
# ⚠️ TASK 92 IS NOT HERE EITHER, AS OF 2026-10-09
#
# installing_a_scanner was dropped for the SAME structural reason as task 0: the
# BDDL goal is `toggled_on` AND `nextto laptop`, and the scanner already starts
# ontop the laptop's table. No transport phase. Full reasoning and the
# "deliberately not backfilled" decision live in src/warm_start.py next to
# CLUSTER_TASKS -- that constant is authoritative, this config follows it.
#
# ---------------------------------------------------------------------------
# ⚠️ THE TASKS ARE NOT EQUALLY REPRESENTED, AND THIS CONFIG DOES NOT EVEN THEM OUT
#
# Batches are sampled uniformly over FRAMES, and the two tasks are different
# lengths:
#
#     77  installing_a_modem          482,597 frames   37.2%
#     89  installing_a_fax_machine    815,378 frames   62.8%
#                                   -----------------
#                                   1,297,975 frames
#
# So task 77 -- the one task we have a single-task reference point for -- still
# gets the LESSER exposure. That is a real asymmetry and it is recorded rather
# than corrected, for two reasons: balancing by dropping episodes throws away
# data we paid to download, and balancing by re-weighting the sampler is a
# change whose effect we could not measure without the single-task comparison we
# deliberately retired. Read per-task results with this table in hand.
#
# ⚠️ Dropping 92 made this asymmetry WORSE per task, not better: 77 went from
# 22.4% to 37.2%, but the gap between the two members is now 25.6 points where
# the widest 3-task gap was 17.2. Task 89's optimum will fall EARLIER than
# task 77's, which is the direct argument for keep_period=2_000 below.
# =============================================================================

from __future__ import annotations

from openpi.training import config as openpi_config
from openpi.training import optimizer as _optimizer
from openpi.training import weight_loaders

# The task-agnostic half of the pipeline: it describes the robot and the dataset
# schema, not the task, so it is shared rather than copied.
from configs.common import (
    NORM_STATS_ASSET_ID,
    NORM_STATS_VENDOR,
    BehaviorDataConfig,
    model_config,
)
from src import warm_start

# PLURAL. Every guard in src/warm_start.py and tests/test_task_configs.py checks
# TASK_INDICES before TASK_INDEX precisely so a multi-task config cannot be
# half-checked -- see normalize_task_indices().
TASK_INDICES = warm_start.CLUSTER_TASKS  # (77, 89)

# The upstream names, in the same order, for the demos directories the converter
# reads. The dense task_index the dataset uses (0, 1) is assigned by the
# converter from the SORTED upstream indices, which is why this tuple is sorted.
TASK_NAMES = ("installing_a_modem", "installing_a_fax_machine")

# The name of the RUN, not of a task. Used for the converted dataset directory,
# the lerobot repo_id and the log header.
TASK_NAME = "cluster_installing_devices"

# --- Where things live on the pod --------------------------------------------
# One converted dataset holding both tasks, built by a single
# convert_v30_to_v21.py invocation with two --src arguments.
DEMOS_DIRS = tuple(f"/workspace/data/demos/{name}" for name in TASK_NAMES)
LEROBOT21_ROOT = "/workspace/data/lerobot21_full"
LEROBOT_REPO_ID = TASK_NAME

# ⚠️ checkpoint_2, BY DECISION -- NOT by lookup.
#
# warm_start_checkpoint() answers "which 2025 checkpoint was trained on this
# task index", and for 77/89/92 the honest answer is "none, they are all >= 50",
# so it falls back to checkpoint_1 for all three -- uniformly, silently, and
# WRONG for this run. checkpoint_1 has never turned anything on. checkpoint_2 is
# the only 2025 checkpoint with any actuation experience (tasks 0 and 8), and
# every task here ends in an actuation.
#
# The cost, recorded: less diversity (16 tasks vs 20) and a slightly weaker
# transport skew (81% vs 90%).
#
# NAMED FALLBACK: if this cluster underperforms specifically on 89 and 92 -- the
# longest transport phases -- suspect THIS CHOICE FIRST and re-run from
# checkpoint_1 as a clean A/B. Anticipated on Day 17; do not re-derive it.
WARM_START = warm_start.CLUSTER_WARM_START  # "checkpoint_2"
CKPT_DIR = f"{warm_start.WINNERS_DIR}/{WARM_START}"

# Their normalisation statistics. Still correct even though checkpoint_2 never
# saw these three tasks, because norm_stats describe the robot's SENSORS, not
# the task. checkpoint_1 and checkpoint_2 ship byte-identical norm_stats.json
# (18,009,212 bytes), which is the empirical proof of exactly that.
NORM_STATS_DIR = f"{CKPT_DIR}/assets/{NORM_STATS_VENDOR}"


def get_config() -> openpi_config.TrainConfig:
    """The training config for the {77, 89, 92} cluster."""
    model = model_config()

    # KNOB -- total training steps.
    #
    # ⚠️ SET HIGH ON PURPOSE, AND STOPPED EARLY BY HAND. The pilot is ~15,000
    # steps (~9 h, ~$7); this says 45,000.
    #
    # That is not a mistake and it is not optimism. openpi derives the entire LR
    # schedule from num_train_steps -- warmup is num_steps // 10 and decay_steps
    # is num_steps -- so running 15k as a COMPLETE cosine cycle and then
    # warm-starting a longer run would restart warmup at 5e-6 -> 5e-5 with fresh
    # Adam state, partly undoing the convergence the first run bought. Setting
    # the budget to what we might actually spend and stopping the pod early
    # keeps every checkpoint on the schedule we meant, and --resume (already
    # wired, train_behavior.py:96,142) continues that same schedule properly.
    #
    # For scale, one full epoch over the cluster is 134,355 steps at batch 16
    # (~82 h, ~$65). 45,000 is 33.5% of an epoch. v1's failure is still a live
    # candidate for UNDERTRAINING -- it saw 9.9% of one epoch -- so always report
    # epochs next to steps and never let a big-looking step count stand in for
    # how much data the model actually saw.
    num_steps = 45_000

    return openpi_config.TrainConfig(
        name="behavior_cluster_installing_devices",
        project_name="behavior-challenge-2026",  # same W&B project, different run
        # Distinct from every other run name. This string decides both the local
        # checkpoint directory and the HuggingFace upload path, so a collision
        # would have one pod overwrite another's run.
        exp_name="cluster_installing_devices_lora_v1",
        model=model,
        data=BehaviorDataConfig(
            repo_id=LEROBOT_REPO_ID,
            base_config=openpi_config.DataConfig(
                # The prompts come from the converted dataset's meta/tasks.jsonl,
                # which now holds THREE rows -- one per task, joined to the
                # frames by task_index. Never hardcoded.
                prompt_from_task=True,
                action_sequence_keys=("action",),
            ),
            assets=openpi_config.AssetsConfig(
                assets_dir=NORM_STATS_DIR,
                asset_id=NORM_STATS_ASSET_ID,
            ),
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader(f"{CKPT_DIR}/params"),
        freeze_filter=model.get_freeze_filter(),
        ema_decay=None,
        batch_size=16,
        num_train_steps=num_steps,
        # Warmup and decay DERIVED from num_steps, never hardcoded. Hardcoding
        # them silently breaks any short run -- at num_steps=1_000 a hardcoded
        # warmup_steps=1_000 makes the entire run a warmup ramp, and the result
        # looks like "the method does not work" when the schedule simply never
        # let the model learn. (Found on p23, 2026-08-15.)
        lr_schedule=_optimizer.CosineDecaySchedule(
            warmup_steps=max(1, num_steps // 10),
            peak_lr=5e-5,
            decay_steps=num_steps,
            decay_lr=5e-6,
        ),
        optimizer=_optimizer.AdamW(clip_gradient_norm=1.0),
        # ⚠️ KEEP THE INTERMEDIATE CHECKPOINTS THIS TIME.
        #
        # v1's 5000-25000 checkpoints were discarded on the reasoning "loss
        # improved monotonically, so there is no overfitting signal" -- which
        # trusted the exact signal that turned out to be meaningless (final loss
        # 0.0133, q_score 0.0). Checkpoints are the only way to find the right
        # step count empirically, and a stopped pod takes its container disk
        # with it.
        #
        # keep_period must divide into a step the run actually reaches: openpi's
        # loop is range(0, num_steps), so the last step is num_steps - 1 and a
        # keep_period equal to num_steps would keep nothing at all.
        #
        # 5_000 -> 2_000, decided 2026-10-09. At a ~20k stop, 5k granularity
        # keeps only 5k/10k/15k/20k, which is coarse enough to straddle a
        # mid-training peak entirely and never see it. The team's own joint
        # 3-task run scored 0.000 at step 8000 and 0.263 at 19999, while
        # single-task 89 peaked at 8000 -- so the peak MOVES with joint
        # training and we cannot predict where it lands. 2_000 divides cleanly
        # into num_steps=45_000 and gives ~10 checkpoints by 20k.
        #
        # This is also the direct consequence of the uneven frame split above:
        # with 77 at 37.2% and 89 at 62.8%, their optima fall at different
        # steps, so there may be NO single best checkpoint for the cluster.
        # Submission is per-task rollouts, so a different checkpoint per task
        # is legitimate -- but only if we KEPT the checkpoints to choose from.
        #
        # ⚠️ Costs ~100 GB of container disk instead of ~30 GB. rebuild_all.sh's
        # GB_OUR_CHECKPOINTS was raised 30 -> 100 to match (2026-10-09). Note
        # the stale figure made the pre-flight too LENIENT, not too strict: it
        # would have passed, started training, and run out of disk hours in.
        # Still ~253 GB total against a 1.4 TB pod, so there is headroom.
        save_interval=1_000,
        keep_period=2_000,
        log_interval=100,
        wandb_enabled=True,
        checkpoint_base_dir="/workspace/checkpoints/ours",
        assets_base_dir="/workspace/assets",
        num_workers=4,
        seed=42,
    )

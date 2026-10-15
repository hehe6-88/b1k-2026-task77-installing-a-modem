# Task 77 `installing_a_modem` — how to load and run our checkpoint

Helen / `hlhl188`. Everything here is for **task 77 only**. Task 89 scored 0.000 at every
checkpoint so we did not submit it — Rena's 89 is the one to use.

```
checkpoint   https://huggingface.co/hlhl188/b1k-2026-cluster-77-89
             runs/cluster_installing_devices_lora_v1/step_20000
score        q = 0.2500 on instances 0–9 (5/10 non-zero, 1 full solve)
```

---

## ⚠️ The one thing that will break your integration

**The LoRA is baked into the model ARCHITECTURE, not shipped as a separate adapter file.**

There is no `adapter_model.safetensors` to merge. The checkpoint's parameter tree already has
the LoRA-shaped tensors in it, so it only loads if you construct `Pi0Config` with the **same
two variants we trained with**:

```python
from openpi.models import pi0_config

model = pi0_config.Pi0Config(
    pi05=True,
    paligemma_variant="gemma_2b_lora",      # <-- NOT the default "gemma_2b"
    action_expert_variant="gemma_300m_lora", # <-- NOT the default "gemma_300m"
    action_dim=32,
    action_horizon=30,
)
```

Load it with the stock `gemma_2b` / `gemma_300m` variants and the parameter shapes will not
match. Depending on the orbax path that is either a loud shape error or a silent partial
restore that runs and produces confident garbage — the second is the dangerous one, so if
you see the policy serving but behaving randomly, check these two strings first.

`action_dim=32` is openpi **padding**; the real action space is 23-wide
(3 base · 4 trunk · 7 left arm · 1 left gripper · 7 right arm · 1 right gripper).

## ✅ What you do NOT need from us

- **Norm stats** — they ship *inside* the checkpoint at
  `step_20000/assets/behavior_224_rgb/norm_stats.json`. Nothing separate to wire up.
- **Robot config** — we used your `r1pro.yaml` unmodified. `r1pro.yaml` is included here only
  so you can diff it; sha256
  `a98fa8a81472adfecfb642778d4d7a01e20131be7f70824e0ae095d665cdbb93`. If that matches yours,
  ignore the file.

## The prompt — must be exact

Read at serve time from the dataset's `meta/tasks.jsonl`, never hardcoded. For task 77 it is
verbatim:

```
Install the modem by placing it on the cabinet and turning it on.
```

A different string puts the policy out of distribution with no error and no log line. This
cost us a whole 30,000-step run once.

---

## Files

| file | what it is | do you need it? |
|---|---|---|
| `serve_behavior_policy.py` | our policy server (websocket + `/healthz`) | only if you don't reuse the shared wrapper |
| `behavior_policy.py` | **`BehaviorInputs` / `BehaviorOutputs`** | **probably yes — see below** |
| `config_cluster_installing_devices.py` | the run config (`TrainConfig`) | yes, or copy the Pi0Config above |
| `common.py` | `ACTION_DIM`, `ACTION_HORIZON`, the data-config factory | imported by the above |
| `task_prompt.py` | reads the prompt from `meta/tasks.jsonl` | only if you want the same guard |
| `r1pro.yaml` | for diffing only | no |

Only `openpi` imports — no dependency on the rest of our private repo.

### The transform question — tell us which way you want it

`behavior_policy.py` does three things to observations/actions that your wrapper may already
do its own way:

1. **State**: re-orders the raw **61-wide** sim state into our **23-wide** vector via
   `ACTION_TO_STATE_INDEX`.
2. **Actions**: we train on **deltas**, not absolutes. Verified empirically — mean
   `|action − next_state|` 0.00275 vs `|action − current|` 0.00325. The delta transform must
   be applied, or the arm will move to the wrong place.
3. **Images**: RGB only, alpha channel dropped (the sim returns RGBA; π0.5 has no depth field).

If the shared wrapper already produces observations in openpi's expected layout, you may only
need the `Pi0Config` above. **If you tell me which observation format your wrapper hands the
policy, I can confirm whether ours is a no-op against it** rather than you having to diff it.

---

## Minimal load, if you just want the policy object

```python
from openpi.policies import policy_config
import config_cluster_installing_devices as behavior_config

cfg = behavior_config.get_config()
policy = policy_config.create_trained_policy(
    cfg,
    "/path/to/step_20000",
    default_prompt="Install the modem by placing it on the cabinet and turning it on.",
)
```

`cfg` carries the Pi0Config, the data transforms and the asset id together, which is why
passing our config is safer than reconstructing the pieces by hand.

⚠️ `create_trained_policy` takes **no seed argument**. Seeding, if you want it, is
`policy._rng = jax.random.key(42)` after construction — and note that does **not** make
rollouts deterministic (Isaac Sim physics and GPU nondeterminism dominate). It makes a run
*auditable*, not reproducible.

## How we ran it, for reference

```bash
# policy server (openpi venv)
python serve_behavior_policy.py --checkpoint <step_20000> \
    --config cluster_installing_devices --task 77 --port 8000 --seed 42
curl -fsS http://127.0.0.1:8000/healthz     # must return 200 before the evaluator

# evaluator (OmniGibson conda env — separate interpreter, conflicting deps)
python -m omnigibson.eval.eval --task-name installing_a_modem \
    --host 127.0.0.1 --port 8000 \
    --instance-indices 0 1 2 3 4 5 6 7 8 9 \
    --num-rollouts 1 --output-dir <out> --write-video --video-fps 30
```

Two interpreters is required, not a workaround — openpi (JAX) and OmniGibson (Isaac Sim) have
genuinely conflicting dependencies. JAX holds ~34 GB, so kill the server between tasks or the
next one OOMs.

**One environment note that cost us a day:** Isaac Sim 5.1's RTX renderer segfaults on NVIDIA
driver **595.x**. Training is unaffected because it never renders — so a machine can train
perfectly for 15 hours and be unable to run a single evaluation. If your eval host is on a
59x driver, that is why. Driver **535–570** works.

Questions → Helen.

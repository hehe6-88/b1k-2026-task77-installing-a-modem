# =============================================================================
# common.py -- The parts of the training config that are the same for EVERY task.
#
# ---------------------------------------------------------------------------
# WHAT BELONGS HERE, AND WHAT DOES NOT
#
# Here: anything that describes the ROBOT or the DATASET SCHEMA. Those are
# fixed across the whole challenge -- the same R1Pro, the same three 224x224
# RGB cameras, the same 23-wide action vector, the same absolute->delta
# conversion. Two drifting copies of the delta-mask derivation would be a real
# bug waiting to happen, so it lives in one place.
#
# NOT here: anything a person would TUNE. Step counts, learning rates, warm
# starts, exp_names, task names. Those live in each task's own folder:
#
#     configs/task23_boxing_books/config.py       task 23  (p23)
#     configs/task77_installing_modem/config.py   task 77  (p77)
#
# That split is the whole point of the folder layout. p23 carries our only
# baseline -- a 30,000-step run and four evaluation rollouts -- so tuning p77
# must not be able to move it. Shared *facts* are safe to share; shared *knobs*
# are not.
#
# Provenance for everything below: the PUBLIC openpi repo
# (Physical-Intelligence/openpi) -- pi0_libero_low_mem_finetune for the LoRA
# pattern, pi05_libero for the pi0.5 pattern, and LeRobotLiberoDataConfig /
# libero_policy.py as the custom-dataset template, which openpi's own comments
# nominate for exactly this purpose.
# =============================================================================

from __future__ import annotations

import dataclasses
import pathlib

from openpi.models import model as _model
from openpi.models import pi0_config
from openpi.training import config as openpi_config
import openpi.transforms as _transforms
from typing_extensions import override

from src import behavior_policy

# The model's internal vector width.
#
# 32, matching the 2025 winners -- and for a reason we derived rather than
# copied. BehaviorInputs re-orders the raw 61-wide state into a 23-wide vector
# aligned with the action vector (see ACTION_TO_STATE_INDEX), so the state
# entering the model is 23 wide, not 61. 32 comfortably clears it.
#
# This supersedes an earlier value of 64, which existed only because we were
# feeding the raw 61-wide state straight through. Once the state is aligned to
# the actions -- which delta conversion requires anyway -- the extra width buys
# nothing and diverges from the checkpoint we warm-start from.
#
# Confirm against a checkpoint's stored shapes with:
#   python scripts/inspect_checkpoint.py <ckpt>
ACTION_DIM = 32

# !! RECONCILE -- action chunk length: how many future timesteps the model
# emits per query. openpi's default is 50; its pi05_libero example uses 10; the
# 2025 winners used 30. At BEHAVIOR's 30 fps, 30 steps is 1 s of motion per
# inference. Not recoverable from the checkpoint's weights -- see
# scripts/inspect_checkpoint.py.
ACTION_HORIZON = 30

# Normalisation statistics live inside the warm-start checkpoint at
#   <checkpoint>/assets/IliaLarchenko/behavior_224_rgb/norm_stats.json
#
# We use THEIRS rather than computing our own, deliberately. Norm stats describe
# the input distribution a model was trained against; warm-starting from their
# weights while normalising differently would feed the model a distribution it
# has never seen. It would load cleanly and train badly.
#
# This stays correct even for a task the winners never trained on (index >= 50),
# because norm_stats describe the ROBOT'S SENSORS, not the task.
#
# Recompute with openpi's scripts/compute_norm_stats.py only if we move to base
# pi0.5, or fine-tune far enough that the distribution genuinely shifts.
NORM_STATS_VENDOR = "IliaLarchenko"
NORM_STATS_ASSET_ID = "behavior_224_rgb"


@dataclasses.dataclass(frozen=True)
class BehaviorDataConfig(openpi_config.DataConfigFactory):
    """Data pipeline for the BEHAVIOR challenge demos.

    Copied from openpi's LeRobotLiberoDataConfig, which its own comments
    nominate as the template for custom datasets, with the transforms swapped
    for ours.
    """

    @override
    def create(
        self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig
    ) -> openpi_config.DataConfig:
        # The repack transform applies to DATASET rows only, never to live
        # inference. Its whole job is to make the dataset's column names look
        # like what the simulator will hand us at eval time.
        #
        # Here it is close to a no-op on names -- our BehaviorInputs reads the
        # dataset's own key names directly -- but the prompt has to be pulled
        # out of the LeRobot task field, which is what PromptFromLeRobotTask
        # does downstream via base_config's prompt_from_task.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        behavior_policy.BASE_IMAGE_KEY: behavior_policy.BASE_IMAGE_KEY,
                        behavior_policy.LEFT_WRIST_IMAGE_KEY: behavior_policy.LEFT_WRIST_IMAGE_KEY,
                        behavior_policy.RIGHT_WRIST_IMAGE_KEY: behavior_policy.RIGHT_WRIST_IMAGE_KEY,
                        "observation.state": "observation.state",
                        "action": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )

        # Applied during BOTH training and inference -- which is the point.
        # Any mismatch between dataset layout and simulator layout is resolved
        # in one place rather than surfacing as a distribution shift at eval.
        data_transforms = _transforms.Group(
            inputs=[
                behavior_policy.BehaviorInputs(
                    model_type=model_config.model_type, action_dim=ACTION_DIM
                )
            ],
            outputs=[behavior_policy.BehaviorOutputs()],
        )

        # Absolute -> delta action conversion.
        #
        # BEHAVIOR's raw actions are ABSOLUTE joint position targets. Verified
        # from meta/stats.json: each arm action dim's distribution matches a
        # measured state dim to 3-4 decimal places, e.g. action[9] and state[5]
        # both have min -2.356, mean -0.048, std 0.425. Deltas would be tightly
        # clustered near zero instead.
        #
        # pi0 models are trained on DELTA actions -- openpi's own comment: "IF
        # your data has absolute actions you can uncomment the following line to
        # convert." So this conversion is required, and its absence was a real
        # bug in the first draft of this config.
        #
        # The mask leaves base velocity and both grippers absolute; see
        # DELTA_MASK_SPEC in behavior_policy.py for the reasoning per dim.
        delta_mask = _transforms.make_bool_mask(*behavior_policy.DELTA_MASK_SPEC)
        data_transforms = data_transforms.push(
            inputs=[_transforms.DeltaActions(delta_mask)],
            outputs=[_transforms.AbsoluteActions(delta_mask)],
        )

        # Tokenises the prompt and the action targets. openpi's comment: "You do
        # not need to change anything here for your own dataset."
        model_transforms = openpi_config.ModelTransformFactory()(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
        )


def model_config() -> pi0_config.Pi0Config:
    """Architecture. Must match the warm-start checkpoint's architecture exactly."""
    return pi0_config.Pi0Config(
        # pi0.5, not pi0. Raises the token budget (max_token_len 200 vs 48) and
        # turns on discrete state input.
        pi05=True,
        # LoRA on both the VLM and the action expert. This is the reason
        # fine-tuning fits in one GPU's memory: we train small adapter matrices
        # rather than all ~3 B parameters, which also stops the pretrained
        # vision/language knowledge being overwritten.
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
        action_dim=ACTION_DIM,
        action_horizon=ACTION_HORIZON,
    )

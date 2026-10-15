# =============================================================================
# behavior_policy.py -- BEHAVIOR-1K <-> pi0.5 input/output transforms.
#
# Modelled on openpi's own template, src/openpi/policies/libero_policy.py,
# which says explicitly: "For your own dataset, you can copy this class and
# modify the keys."  This is that copy, for the BEHAVIOR R1Pro robot.
#
# Written 2026-08-10 (Day 10), from the public openpi repo and the dataset's
# own meta/info.json. Not derived from any other competitor's code.
#
# ---------------------------------------------------------------------------
# WHAT THESE DO
#
# BehaviorInputs   runs during TRAINING and INFERENCE. It takes a row of the
#                  dataset (or a live observation from the simulator) and packs
#                  it into the exact dict pi0.5 expects.
#
# BehaviorOutputs  runs during INFERENCE only. The model emits action vectors
#                  padded out to its internal width; this trims them back to
#                  the 23 numbers the R1Pro actually accepts.
#
# The pairing matters: if inputs pad and outputs do not trim, the robot gets
# handed a 32- or 64-wide vector it cannot interpret.
# =============================================================================

from __future__ import annotations

import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model

# --- Measured facts about the BEHAVIOR data ----------------------------------
# From meta/info.json of behavior-1k/2026-challenge-demos. Measurements, not
# choices -- do not "tune" these.

BEHAVIOR_ACTION_DIM = 23  # action, float32, shape [23]
BEHAVIOR_STATE_DIM = 61  # observation.state, float32, shape [61]

# --- The action-aligned state map --------------------------------------------
#
# WHY THIS EXISTS
#
# openpi's DeltaActions computes `actions[i] -= state[i]` -- positionally. That
# is only correct if action dim i and state dim i are the same joint. In the
# raw BEHAVIOR data they are NOT: actions are 23 wide, state is 61 wide, and
# the correspondence is scattered. So before any delta conversion we rebuild a
# 23-wide state whose entries line up with the action vector.
#
# HOW THE MAP WAS DERIVED
#
# From meta/stats.json, by matching each action dim's (min, mean, std) against
# every state dim's. The arm mappings are unambiguous -- distributions agree to
# 3-4 decimal places, e.g.
#
#     action[9]  min -2.356  mean -0.048  std 0.425
#     state[5]   min -2.356  mean -0.048  std 0.425
#
# That is not a coincidence; it means actions are ABSOLUTE joint position
# targets living in the same space as the measured positions.
#
# The full 61-dim layout this implies:
#     0-2    base velocity (robot-local frame; confirmed by the 2026-07-27 update)
#     3-9    left arm positions        10-16  left arm velocities
#     17-23  left end-effector pose    24-25  left gripper fingers
#     26-27  left finger velocities
#     28-34  right arm positions       35-41  right arm velocities
#     42-48  right end-effector pose   49-50  right gripper fingers
#     51-52  right finger velocities
#     53-55  torso positions           56-60  torso extras / velocities
#
# CONFIDENCE
#   arms (7+7 dims)  CERTAIN -- exact distribution match
#   torso (3 dims)   CERTAIN -- exact distribution match
#   base (3 dims)    HIGH -- corroborated by the organisers' 07/27 note
#   grippers         INFERRED -- action is normalised [-1, 1] while state is
#                    finger position [0, 0.05], so they cannot be matched by
#                    distribution. Harmless for delta conversion, which leaves
#                    grippers absolute anyway (see DELTA_MASK).
#
# Verify against real data with scripts/verify_state_map.py before trusting it
# for a full training run.
ACTION_TO_STATE_INDEX = (
    0, 1, 2,                      # 0-2   base velocity  (x, y, yaw)
    53, 54, 55,                   # 3-5   torso positions
    56,                           # 6     dead dim -- action is always exactly 0
    3, 4, 5, 6, 7, 8, 9,          # 7-13  left arm positions
    24,                           # 14    left gripper
    28, 29, 30, 31, 32, 33, 34,   # 15-21 right arm positions
    49,                           # 22    right gripper
)
assert len(ACTION_TO_STATE_INDEX) == BEHAVIOR_ACTION_DIM

# Which action dims get converted to deltas (True) and which stay absolute
# (False). Passed to openpi's DeltaActions/AbsoluteActions pair.
#
#   base velocity  absolute -- it is already a rate, not a position; subtracting
#                  the current velocity from a velocity command is meaningless
#   torso, arms    delta -- absolute position targets, which is exactly what the
#                  delta transform is for
#   dead dim 6     absolute -- always 0; nothing to convert
#   grippers       absolute -- openpi keeps grippers absolute (see its Libero
#                  example), and ours are on a different scale from the finger
#                  state anyway
#
# Layout: 3 base | 3 torso | 1 dead | 7 left arm | 1 grip | 7 right arm | 1 grip
DELTA_MASK_SPEC = (-3, 3, -1, 7, -1, 7, -1)  # -> make_bool_mask(*DELTA_MASK_SPEC)

# The three RGB streams the dataset ships, in pi0.5's slot order. The ZED is
# the head camera so it plays the third-person "base" role; the two RealSense
# cameras are wrist-mounted.
#
# There are also three depth_linear streams. We ignore them: pi0.5's
# Observation has no depth field, so they could not be consumed even if we
# downloaded them (and skipping them halves the video bytes).
BASE_IMAGE_KEY = "observation.rgb.zed_link_camera_0"
LEFT_WRIST_IMAGE_KEY = "observation.rgb.left_realsense_link_camera_0"
RIGHT_WRIST_IMAGE_KEY = "observation.rgb.right_realsense_link_camera_0"


def _parse_image(image) -> np.ndarray:
    """Coerce an image to uint8 HWC.

    LeRobot hands back float32 CHW; the simulator hands back uint8 HWC. The
    model wants uint8 HWC either way, so normalise here rather than making
    every caller remember which source it is dealing with.
    """
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # CHW -> HWC
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class BehaviorInputs(transforms.DataTransformFn):
    """Pack a BEHAVIOR observation into pi0.5's expected input dict.

    Used for BOTH training and inference, which is the point: whatever shape
    mismatch exists between the dataset and the simulator gets resolved here,
    once, instead of appearing as a mysterious distribution shift at eval time.
    """

    # Set from the model config; decides how padded images are masked.
    model_type: _model.ModelType

    # The model's internal vector width. Both `state` and `actions` are padded
    # up to this.
    #
    # pi0 sizes the state vector to action_dim -- in pi0_config.py the state
    # spec is [batch, action_dim]. BEHAVIOR's state is 61 wide and its actions
    # are 23, so with stock Pi0Config this needs to be >= 61.
    #
    # Note openpi's pad_to_dim only ever GROWS an array -- if the state is
    # already wider than action_dim it is returned untouched, not truncated. So
    # too small a value here does not silently lose proprioception; it produces
    # a shape mismatch when the state projection runs. Loud, not silent.
    #
    # !! Still must match the checkpoint we warm-start from. The 2025 winners
    # used action_dim=32, but with a custom model config and a 256-wide state,
    # so their number does not transfer to this setup unexamined.
    action_dim: int = 64

    def __call__(self, data: dict) -> dict:
        # All three cameras genuinely exist on the R1Pro, so unlike the Libero
        # example we never have to substitute a zero image. All three masks are
        # therefore True.
        base_image = _parse_image(data[BASE_IMAGE_KEY])
        left_wrist = _parse_image(data[LEFT_WRIST_IMAGE_KEY])
        right_wrist = _parse_image(data[RIGHT_WRIST_IMAGE_KEY])

        state = np.asarray(data["observation.state"], dtype=np.float32)
        if state.shape[-1] != BEHAVIOR_STATE_DIM:
            raise ValueError(
                f"expected {BEHAVIOR_STATE_DIM}-dim observation.state for R1Pro, "
                f"got {state.shape[-1]}. The dataset layout may have changed."
            )

        # Re-order the 61-wide state into a 23-wide vector aligned with the
        # action vector, so that state[i] and action[i] are the same joint.
        #
        # This is load-bearing, not tidiness: openpi's DeltaActions subtracts
        # state[i] from action[i] positionally, so without this the delta
        # conversion would subtract unrelated joints from each other and
        # produce targets that look plausible and are meaningless.
        state = np.take(state, ACTION_TO_STATE_INDEX, axis=-1)

        # Now only 23 wide, so it pads rather than overflows -- which is also
        # why action_dim=32 is sufficient despite the raw state being 61.
        state = transforms.pad_to_dim(state, self.action_dim)

        inputs = {
            "state": state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist,
                "right_wrist_0_rgb": right_wrist,
            },
            # Every camera is real, so nothing is masked out.
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }

        # Actions exist during training, not during inference.
        if "action" in data:
            actions = np.asarray(data["action"], dtype=np.float32)
            inputs["actions"] = transforms.pad_to_dim(actions, self.action_dim)

        # The language instruction. pi0.5 is a vision-LANGUAGE-action model, so
        # this is a real input. The challenge dataset ships a natural-language
        # string per task; it is resolved from the dataset rather than written
        # down anywhere in our code -- see src/task_prompt.py for why.
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class BehaviorOutputs(transforms.DataTransformFn):
    """Trim the model's padded action vector back to the robot's 23 dims.

    Inference only. Must stay paired with BehaviorInputs' padding: the model
    emits action_dim numbers, of which only the first 23 mean anything.
    """

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :BEHAVIOR_ACTION_DIM])}


def make_behavior_example() -> dict:
    """A random input of the right shapes, for smoke-testing the pipeline.

    Lets us exercise the transforms without any data on disk -- useful for
    catching shape errors on the laptop before renting a GPU.
    """
    return {
        BASE_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        LEFT_WRIST_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        RIGHT_WRIST_IMAGE_KEY: np.random.randint(256, size=(224, 224, 3), dtype=np.uint8),
        "observation.state": np.random.rand(BEHAVIOR_STATE_DIM).astype(np.float32),
        "action": np.random.rand(BEHAVIOR_ACTION_DIM).astype(np.float32),
        # Deliberately NOT the real task description. This example only exercises
        # tensor shapes, and a realistic-looking string here reads as
        # authoritative and gets copied into places where it matters. The real
        # prompt is resolved from the dataset -- see src/task_prompt.py.
        "prompt": "smoke-test prompt, not the real task description",
    }

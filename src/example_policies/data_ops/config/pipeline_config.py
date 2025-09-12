# Copyright 2025 Poke & Wiggle GmbH. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import asdict, dataclass

# String Enum for action levels
from enum import Enum
from typing import Any, Dict


class ActionLevel(Enum):
    JOINT = "joint"
    TCP = "tcp"
    TELEOP = "teleop"
    DELTA_TCP = "delta_tcp"
    DELTA_JOINT = "delta_joint"


class GripperType(Enum):
    PANDA = "panda"
    ROBOTIQ = "robotiq"


@dataclass
class PipelineConfig:
    """Configuration for what data to include in the dataset.

    Action representation semantics (action_level):
      - TCP / TELEOP:
          Action = [ left_xyz(3), left_quat_xyzw(4),
                     right_xyz(3), right_quat_xyzw(4),
                     gripper_left(1), gripper_right(1) ]
          Dim = 16 (assuming 1 scalar per gripper). Quaternions are expected normalized (unit norm).
      - DELTA_TCP:
          Action = [ d_left_xyz(3), d_left_rotvec(3),
                     d_right_xyz(3), d_right_rotvec(3),
                     gripper_left(1), gripper_right(1) ]
          Dim = 14. rotvec = axis * angle (radians), magnitude = rotation angle. Integration downstream
          must consistently apply either left-multiplication on SO(3) via exp(rotvec) ⊗ quat_prev or
          right-multiplication; mixing conventions degrades policy quality.
      - JOINT:
          Action = [ left_joint_0..6 (7), right_joint_0..6 (7), gripper_left(1), gripper_right(1) ]
          Dim = 16. Absolute target joint angles (radians).
      - DELTA_JOINT:
          Same layout as JOINT but values are additive deltas to be applied to previous commanded or
          measured joint positions. Units: radians.

    Args:
        include_joint_positions: Include 7 left + 7 right joint angles in observation.state.
        include_joint_velocities: Include joint velocities (same ordering as positions).
        include_joint_efforts: Include joint efforts/torques.
        include_tcp_poses: Include current TCP pose (xyz + quat) for each arm.
        include_last_command: Include last commanded TCP pose (temporal context for delta policies).
        include_rgb_images: Record left/right RGB streams (and static camera).
        include_depth_images: Record left/right depth streams (same resolution).
        image_resolution: (width, height) for all image modalities (uniform assumption).
        depth_scale_factor: Scalar to convert depth units (e.g. meters -> millimeters) during storage.
        action_level: Enum selecting action representation (see above).
        left_gripper: GripperType for left arm (affects observation.state length).
        right_gripper: GripperType for right arm.
        task_name: Human-readable task descriptor for metadata.
        max_pause_seconds: Threshold to classify inactivity (combined with pause_velocity).
        pause_velocity: Norm velocity below which motion considered paused.
        save_pauses: If True, saves pauses into an extra dataset for validation.
        gripper_active_speed: Velocity threshold to consider gripper in active motion (for speedup logic).
        boost_factor: Temporal acceleration factor for segments (if implemented externally).
        grace_period_seconds: Hysteresis window to avoid rapid pause/unpause flipping.
        save_normal: If True, always save non-boosted stream (dual outputs) for validation.
        recording_fps: Raw acquisition frequency from robot / sensors.
        target_fps: Desired resulting FPS after subsampling.
        subsample_offset: Phase offset into frame sequence when subsampling.
        min_episode_seconds: Minimum episode duration retained.

    """

    include_joint_positions: bool = False
    include_joint_velocities: bool = False
    include_joint_efforts: bool = False
    include_tcp_poses: bool = True
    include_last_command: bool = True

    include_rgb_images: bool = True
    include_depth_images: bool = False
    image_resolution = (640, 640)
    depth_scale_factor: float = 1000.0

    action_level: ActionLevel = ActionLevel.DELTA_TCP

    # Gripper type
    left_gripper: GripperType = GripperType.PANDA
    right_gripper: GripperType = GripperType.PANDA

    # Task name
    task_name: str = "Push the box"

    # Pauses
    max_pause_seconds: float = 0.2
    pause_velocity: float = 0.03
    save_pauses: bool = False

    # Speedup
    gripper_active_speed: float = 0.05
    boost_factor: int = 1
    grace_period_seconds: float = 0.2
    save_normal: bool = False

    # Subsampling
    recording_fps: int = 30
    target_fps: int = 10
    subsample_offset: int = 0

    min_episode_seconds: int = 15

    def __post_init__(self):
        self.include_joint_states = (
            self.include_joint_positions
            or self.include_joint_velocities
            or self.include_joint_efforts
        )

        self.include_images = self.include_rgb_images or self.include_depth_images

        # Set image shapes from resolution
        w, h = self.image_resolution
        self.static_cam_shape = (h, w, 3)
        self.rgb_shape = (h, w, 3)
        self.depth_shape = (h, w, 3)

        # Calculate Settings from Subsampling
        self.capture_frequency = int(self.recording_fps / self.target_fps)
        self.min_episode_frames = self.min_episode_seconds * self.target_fps

        self.max_pause_frames = self.max_pause_seconds * self.target_fps
        self.grace_period_frames = self.grace_period_seconds * self.target_fps

    def to_dict(self):
        def convert_enums(obj):
            if isinstance(obj, dict):
                return {k: convert_enums(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_enums(item) for item in obj]
            elif isinstance(obj, Enum):
                return obj.value
            else:
                return obj

        return convert_enums(asdict(self))


def build_features(config: PipelineConfig) -> Dict[str, Any]:
    """Build features dictionary based on configuration."""
    features = {}

    # Build observation state features
    state_names = []

    if config.include_joint_positions:
        # left + right joint positions
        state_names.extend([f"joint_pos_left_{i}" for i in range(7)])
        state_names.extend([f"joint_pos_right_{i}" for i in range(7)])

    if config.include_joint_velocities:
        # left + right joint velocities
        state_names.extend([f"joint_vel_left_{i}" for i in range(7)])
        state_names.extend([f"joint_vel_right_{i}" for i in range(7)])

    if config.include_joint_efforts:
        state_names.extend([f"joint_eff_left_{i}" for i in range(7)])
        state_names.extend([f"joint_eff_right_{i}" for i in range(7)])

    if config.include_tcp_poses:
        # left + right TCP poses
        state_names.extend([f"tcp_left_pos_{i}" for i in "xyz"])
        state_names.extend([f"tcp_left_quat_{i}" for i in "xyzw"])
        state_names.extend([f"tcp_right_pos_{i}" for i in "xyz"])
        state_names.extend([f"tcp_right_quat_{i}" for i in "xyzw"])

    if config.left_gripper == GripperType.PANDA:
        state_names.extend([f"gripper_left_{i}" for i in range(2)])
    elif config.left_gripper == GripperType.ROBOTIQ:
        state_names.extend([f"robotiq_left_{i}" for i in range(6)])
    else:
        raise NotImplementedError(f"Unsupported gripper type {config.left_gripper}")

    if config.right_gripper == GripperType.PANDA:
        state_names.extend([f"gripper_right_{i}" for i in range(2)])
    elif config.right_gripper == GripperType.ROBOTIQ:
        state_names.extend([f"robotiq_right_{i}" for i in range(6)])
    else:
        raise NotImplementedError(f"Unsupported gripper type {config.right_gripper}")

    if config.include_last_command:
        # left + right last command
        state_names.extend([f"last_command_left_{i}" for i in range(7)])
        state_names.extend([f"last_command_right_{i}" for i in range(7)])

    features["observation.state"] = {
        "dtype": "float32",
        "shape": (len(state_names),),
        "names": state_names,
    }

    # Build action features (always TCP poses for now)
    if config.action_level in [ActionLevel.TCP, ActionLevel.TELEOP]:
        names = [f"tcp_left_{i}" for i in "xyz"]
        names += [f"tcp_left_quat_{i}" for i in "xyzw"]
        names += [f"tcp_right_{i}" for i in "xyz"]
        names += [f"tcp_right_quat_{i}" for i in "xyzw"]
    elif config.action_level == ActionLevel.DELTA_TCP:
        names = [f"delta_tcp_left_{i}" for i in "xyz"]
        names += [f"delta_tcp_left_rot_{i}" for i in "xyz"]
        names += [f"delta_tcp_right_{i}" for i in "xyz"]
        names += [f"delta_tcp_right_rot_{i}" for i in "xyz"]
    elif config.action_level in [ActionLevel.JOINT, ActionLevel.DELTA_JOINT]:
        prefix = ""
        if config.action_level == ActionLevel.DELTA_JOINT:
            prefix = "delta_"
        names = [f"{prefix}joint_left_{i}" for i in range(7)]
        names += [f"{prefix}joint_right_{i}" for i in range(7)]
    else:
        raise NotImplementedError(f"Unsupported action level {config.action_level}")
    names += ["gripper_left"]
    names += ["gripper_right"]
    features["action"] = {"dtype": "float32", "shape": (len(names),), "names": names}

    # Build image features
    features["observation.images.rgb_static"] = {
        "dtype": "video",
        "shape": list(config.static_cam_shape),
        "names": ["height", "width", "channel"],
    }

    if config.include_rgb_images:
        features["observation.images.rgb_left"] = {
            "dtype": "video",
            "shape": list(config.rgb_shape),
            "names": ["height", "width", "channel"],
        }
        features["observation.images.rgb_right"] = {
            "dtype": "video",
            "shape": list(config.rgb_shape),
            "names": ["height", "width", "channel"],
        }

    if config.include_depth_images:
        features["observation.images.depth_left"] = {
            "dtype": "video",
            "shape": list(config.depth_shape),
            "names": ["height", "width", "channel"],
        }
        features["observation.images.depth_right"] = {
            "dtype": "video",
            "shape": list(config.depth_shape),
            "names": ["height", "width", "channel"],
        }

    return features


def create_config_from_args(args) -> PipelineConfig:
    """Create DataConfig from command line arguments."""
    return PipelineConfig(
        include_joint_positions=args.include_joint_positions,
        include_joint_velocities=args.include_joint_velocities,
        include_joint_efforts=args.include_joint_efforts,
        include_tcp_poses=args.include_tcp_poses,
        include_rgb_images=args.include_rgb_images,
        include_depth_images=args.include_depth_images,
        action_level=ActionLevel(args.action_level),
        task_name=args.task_name,
    )

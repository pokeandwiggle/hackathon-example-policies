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

import numpy as np
import torch

from example_policies.data_ops.utils import geometric, image_processor
from example_policies.data_ops.utils.message_parsers import CANONICAL_ARM_JOINTS


class ObservationBuilder:
    def __init__(self, cfg):
        self.cfg = cfg

        # Set some default expectations
        self.include_joint_pos = False
        self.include_joint_vel = False
        self.include_joint_effort = False

        self.include_tcp = True
        self.include_last_commands = True

        self.state_feature_names: list[str] = []

        self.configure_metadata(cfg)

    def configure_metadata(self, cfg):
        # Legacy Checkpoints / Lerobot Checkpoints are saved without metadata
        if not cfg.metadata:
            return

        self.state_feature_names = cfg.metadata["features"]["observation.state"][
            "names"
        ]
        names = self.state_feature_names

        # Check if any state feature contains "joint_pos_"
        self.include_joint_pos = any("joint_pos_" in name for name in names)
        self.include_joint_vel = any("joint_vel_" in name for name in names)
        self.include_joint_effort = any("joint_eff_" in name for name in names)

        self.include_tcp = any("tcp_" in name for name in names)

        self.include_last_commands = any("last_command_" in name for name in names)

        # TODO: Detect Gripper Schema

    @property
    def include_joint_state(self):
        return (
            self.include_joint_pos
            or self.include_joint_vel
            or self.include_joint_effort
        )

    def get_observation(self, snapshot_response, last_command, device):
        observation = {}
        if not snapshot_response.robots:
            print("No robots found in snapshot")
            return None
        robot_names = ["left", "right"]

        joint_state = self._get_joint_state(snapshot_response)
        tcp_state = self._get_tcp_state(snapshot_response, robot_names)
        gripper_state = self._get_gripper_state(snapshot_response)
        last_command = self._get_last_command(tcp_state, last_command)

        state_array = []
        if self.include_joint_state:
            state_array.append(joint_state)
        if self.include_tcp:
            state_array.append(tcp_state)
        state_array.append(gripper_state)
        if self.include_last_commands:
            state_array.append(last_command)

        full_state = np.concatenate(state_array).astype(np.float32)

        assert full_state.shape == self.cfg.input_features["observation.state"].shape, (
            f"Observation Builder State shape mismatch: expected {self.cfg.input_features['observation.state'].shape}, "
            f"got {full_state.shape}"
        )

        observation["observation.state"] = (
            torch.from_numpy(full_state).to(device).unsqueeze(0)
        )

        images = self.process_all_cameras(snapshot_response, device)
        observation.update(images)

        return observation

    def process_all_cameras(self, snapshot_response, device):
        images = {}

        cameras = snapshot_response.cameras
        # ['cam_right_color_optical_frame', 'cam_right_depth_optical_frame', 'cam_static_optical_frame', 'cam_left_depth_optical_frame', 'cam_left_color_optical_frame']
        for cam_name in list(cameras.keys()):
            images.update(
                self._process_camera_image(cameras[cam_name], cam_name, device)
            )
        return images

    def _get_joint_state(self, snapshot) -> np.ndarray:
        """Extracts joint states from the snapshot for the given robots."""
        joint_positions = []
        joint_velocities = []
        joint_efforts = []

        for j in CANONICAL_ARM_JOINTS:
            if self.include_joint_pos:
                joint_positions.append(snapshot.joints[j].position)
            if self.include_joint_vel:
                joint_velocities.append(snapshot.joints[j].velocity)
            if self.include_joint_effort:
                joint_efforts.append(snapshot.joints[j].effort)

        return np.array(
            np.concatenate([joint_positions, joint_velocities, joint_efforts]),
            dtype=np.float32,
        )

    def _get_tcp_state(self, snapshot, robot_names) -> np.ndarray:
        """Extracts TCP poses from the snapshot for the given robots."""
        tcp_poses = []
        for robot_name in robot_names:
            pose = snapshot.robots[robot_name].pose
            pose_arr = np.array(
                [
                    pose.position.x,
                    pose.position.y,
                    pose.position.z,
                    pose.orientation.x,
                    pose.orientation.y,
                    pose.orientation.z,
                    pose.orientation.w,
                ]
            )
            pose_arr = geometric.positive_quat(pose_arr)
            tcp_poses.append(pose_arr)

        return np.array(tcp_poses, dtype=np.float32).flatten()

    def _get_gripper_state(self, snapshot) -> np.ndarray:
        """Extracts gripper joint positions from the snapshot."""
        # The order here must match the training data's state representation
        gripper_joint_names = [
            "panda_left_finger_joint1",
            "panda_left_finger_joint2",
            "panda_right_finger_joint1",
            "panda_right_finger_joint2",
        ]
        gripper_positions = [
            snapshot.joints[name].position for name in gripper_joint_names
        ]
        return np.array(gripper_positions, dtype=np.float32)

    def _get_last_command(self, tcp_state: np.ndarray, last_command) -> np.ndarray:
        cmd = []
        if self.include_last_commands:
            if last_command is None:
                cmd = tcp_state
            else:
                cmd = last_command
        return np.array(cmd, dtype=np.float32)

    def _process_camera_image(self, camera_data, camera_name: str, device):
        """Processes a batch of camera images and returns them in a dict."""
        cfg = self.cfg

        side = camera_name.split("_")[1]
        modality = "depth" if "depth" in camera_name else "rgb"

        obs_key = f"observation.images.{modality}_{side}"
        if obs_key not in cfg.input_features:
            print(f"WARN: Observation key {obs_key} not in model input features.")
            return {}

        # Lerobot Model Config Shape is Channel Height Width for some reason
        cfg_shape = cfg.input_features[obs_key].shape

        img_array = image_processor.process_image_bytes(
            camera_data.data,
            width=cfg_shape[2],
            height=cfg_shape[1],
            is_depth=(modality == "depth"),
        )
        image = torch.from_numpy(img_array)
        image = image.permute(2, 0, 1)
        image = image.to(device)
        return {obs_key: image.unsqueeze(0)}

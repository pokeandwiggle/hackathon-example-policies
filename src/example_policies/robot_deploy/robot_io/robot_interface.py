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

from ...utils.action_order import (
    GET_LEFT_GRIPPER_IDX,
    GET_RIGHT_GRIPPER_IDX,
    LEFT_ARM,
    RIGHT_ARM,
    ActionMode,
)
from ...utils.state_order import CANONICAL_ARM_JOINTS
from .observation_builder import ObservationBuilder
from .robot_client import RobotClient
from .robot_service import robot_service_pb2, robot_service_pb2_grpc


class RobotInterface:
    """Handles communication and data conversion with the robot gRPC service."""

    def __init__(self, service_stub: robot_service_pb2_grpc.RobotServiceStub, cfg):
        self.client = RobotClient(service_stub)
        self.observation_builder = ObservationBuilder(cfg)
        self.robot_names = None
        self.last_command = None

    def get_observation(self, device):
        """Gets the current observation from the robot."""
        snapshot_response, self.robot_names = self.client.get_snapshot()

        obs = self.observation_builder.get_observation(
            snapshot_response, self.last_command, device
        )
        return obs

    def send_action(
        self,
        action: torch.Tensor,
        action_mode: ActionMode,
        ctrl_mode: str = RobotClient.CART_QUEUE,
    ):
        """Sends a predicted action to the robot service."""
        numpy_action = action.squeeze(0).to("cpu").numpy()
        # Last Command is always stored in ABS Format
        self.last_command = numpy_action[
            : GET_LEFT_GRIPPER_IDX(
                ActionMode.get_absolute_mode(action_mode=action_mode)
            )
        ]

        if action_mode in (ActionMode.DELTA_TCP, ActionMode.TCP, ActionMode.TELEOP):
            target = _build_cart_target(numpy_action, action_mode)

            if ctrl_mode == RobotClient.CART_DIRECT:
                self.client.send_cart_direct_target(target)
            elif ctrl_mode == RobotClient.CART_WAYPOINT:
                self.client.send_cart_waypoint(target)
            elif ctrl_mode == RobotClient.CART_QUEUE:
                self.client.send_cart_queue_target(target)
            else:
                raise RuntimeError(f"Unknown ctrl_mode: {ctrl_mode}")

        elif action_mode in (ActionMode.DELTA_JOINT, ActionMode.JOINT):
            target = _build_joint_target(numpy_action, action_mode)
            self.client.send_joint_direct_target(target)
        else:
            raise RuntimeError(f"Unknown action mode: {action_mode}")

    def move_home(self):
        """Sends a command to move the robot to its home position."""
        self.client.send_move_home()


def _build_des_pose_msg(action_slice: np.ndarray) -> robot_service_pb2.Pose:
    """Creates a RobotDesired message from an action slice."""
    des_pose = robot_service_pb2.Pose()

    des_pose.position.x = action_slice[0]
    des_pose.position.y = action_slice[1]
    des_pose.position.z = action_slice[2]
    des_pose.orientation.x = action_slice[3]
    des_pose.orientation.y = action_slice[4]
    des_pose.orientation.z = action_slice[5]
    des_pose.orientation.w = action_slice[6]

    return des_pose


def _build_cart_target(
    np_action: np.ndarray, action_mode: ActionMode
) -> robot_service_pb2.CartesianTarget:
    left_desired = _build_des_pose_msg(np_action[LEFT_ARM])
    left_gripper = np_action[GET_LEFT_GRIPPER_IDX(action_mode)]
    right_desired = _build_des_pose_msg(np_action[RIGHT_ARM])
    right_gripper = np_action[GET_RIGHT_GRIPPER_IDX(action_mode)]

    des_target_msg = robot_service_pb2.CartesianTarget()
    des_target_msg.robot_poses["left"].CopyFrom(left_desired)
    des_target_msg.robot_poses["right"].CopyFrom(right_desired)

    des_target_msg.gripper_widths["left"] = left_gripper
    des_target_msg.gripper_widths["right"] = right_gripper

    des_target_msg.robot_stiffness_factors["left"] = 1.0
    des_target_msg.robot_stiffness_factors["right"] = 1.0

    return des_target_msg


def _build_joint_target(np_action: np.ndarray, action_mode: ActionMode
) -> robot_service_pb2.JointTarget:
    """Creates a RobotDesired message from an action slice."""
    des_target_msg = robot_service_pb2.JointTarget()
    for i, joint_name in enumerate(CANONICAL_ARM_JOINTS):
        des_target_msg.joint_angles[joint_name] = np_action[i]

    des_target_msg.gripper_widths["left"] = np_action[
        GET_LEFT_GRIPPER_IDX(action_mode)
    ]
    des_target_msg.gripper_widths["right"] = np_action[
        GET_RIGHT_GRIPPER_IDX(action_mode)
    ]

    des_target_msg.robot_stiffness_factors["left"] = 1.0
    des_target_msg.robot_stiffness_factors["right"] = 1.0

    return des_target_msg

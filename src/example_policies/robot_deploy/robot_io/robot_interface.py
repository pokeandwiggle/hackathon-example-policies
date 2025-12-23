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
        """Sends a command to move the robot to its home position and opens the grippers."""
        # Move home
        try:
            self.client.send_move_home()
        except Exception as e:
            # If homing fails due to controller state, try recovery and retry
            error_msg = str(e).lower()
            if "trajectory controller" in error_msg or "controller" in error_msg:
                print("Homing failed, attempting controller recovery...")
                try:
                    recover_request = robot_service_pb2.RecoverErrorsRequest()
                    self.client.stub.RecoverErrors(recover_request)
                    print("Recovery complete, retrying home...")
                    self.client.send_move_home()
                except Exception as recovery_error:
                    print(f"Recovery failed: {recovery_error}")
                    raise  # Re-raise the original error
            else:
                raise  # Re-raise if it's a different error
        
        # Open grippers
        for _ in range(3):  # Minimum 3 messages required by stability buffer (STABILITY_BUFFER_SIZE)
            gripper_target = robot_service_pb2.JointTarget()
            
            # Set gripper widths to open (normalized 0.0-1.0, where 1.0 = fully open)
            gripper_target.gripper_widths["left"] = 1.0
            gripper_target.gripper_widths["right"] = 1.0
            
            # Send directly via gRPC without changing control mode
            set_target_request = robot_service_pb2.SetJointTargetRequest()
            set_target_request.joint_target.CopyFrom(gripper_target)
            self.client.stub.SetJointTarget(set_target_request)
            
    def move_to_joint_goal(
        self, joint_angles: np.ndarray, joint_names: list[str] = CANONICAL_ARM_JOINTS
    ):
        """Moves the robot to a specific joint configuration using trajectory planning.

        Args:
            joint_angles: Array of target joint angles (radians)
            joint_names: List of joint names corresponding to the angles.
                        Defaults to CANONICAL_ARM_JOINTS (left + right arm joints)

        Returns:
            The response from the MoveToJointGoal gRPC call

        Raises:
            Exception: If the move fails
            ValueError: If joint_angles and joint_names have different lengths
        """
        return self.client.move_to_joint_goal(joint_angles, joint_names)

    def set_gripper_state(
        self, gripper_id: str, width: float, speed: float = 0.0, force: float = 0.0
    ):
        """Controls the gripper state (open/close).

        Args:
            gripper_id: Gripper identifier ("left" or "right")
            width: Desired gripper width in meters
            speed: Gripper speed in m/s
            force: Gripper force in Newtons

        Returns:
            The response from the SetGripperState gRPC call
        """
        return self.client.set_gripper_state(gripper_id, width, speed, force)

    def visualize_action(
        self,
        action: torch.Tensor,
        action_mode: ActionMode,
    ):
        """Visualizes a trajectory represented as a tensor of actions.

        For Cartesian modes (DELTA_TCP, TCP, TELEOP), creates a sequence of
        Cartesian targets and visualizes them. For joint modes (DELTA_JOINT, JOINT),
        creates a sequence of joint targets and visualizes them.

        Args:
            action: Tensor of actions to visualize. Expected shape: (sequence_length, action_dim)
            action_mode: The mode of the action (determines how to interpret the tensor)
        """
        numpy_action = action.to("cpu").numpy()

        # Handle both 2D (sequence_length, action_dim) and 3D (batch, sequence_length, action_dim) tensors
        if numpy_action.ndim == 3:
            numpy_action = numpy_action.squeeze(0)

        if action_mode in (ActionMode.DELTA_TCP, ActionMode.TCP, ActionMode.TELEOP):
            # Build Cartesian targets for each step in the trajectory
            cart_targets = []
            for action_step in numpy_action:
                target = _build_cart_target(action_step, action_mode)
                cart_targets.append(target)

            # Visualize the trajectory
            self.client.visualize_cart_direct_targets(cart_targets)

        elif action_mode in (ActionMode.DELTA_JOINT, ActionMode.JOINT):
            # Build joint targets for each step in the trajectory
            joint_targets = []
            for action_step in numpy_action:
                target = _build_joint_target(action_step, action_mode)
                joint_targets.append(target)

            # Visualize the trajectory
            self.client.visualize_joint_direct_targets(joint_targets)
        else:
            raise RuntimeError(f"Unknown action mode: {action_mode}")


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
    # Action is always in absolute format after action_translator, so use absolute indices for grippers
    abs_mode = ActionMode.get_absolute_mode(action_mode)
    left_desired = _build_des_pose_msg(np_action[LEFT_ARM])
    left_gripper = np_action[GET_LEFT_GRIPPER_IDX(abs_mode)]
    right_desired = _build_des_pose_msg(np_action[RIGHT_ARM])
    right_gripper = np_action[GET_RIGHT_GRIPPER_IDX(abs_mode)]

    des_target_msg = robot_service_pb2.CartesianTarget()
    des_target_msg.robot_poses["left"].CopyFrom(left_desired)
    des_target_msg.robot_poses["right"].CopyFrom(right_desired)

    des_target_msg.gripper_widths["left"] = left_gripper
    des_target_msg.gripper_widths["right"] = right_gripper

    des_target_msg.robot_stiffness_factors["left"] = 0.7
    des_target_msg.robot_stiffness_factors["right"] = 0.7

    return des_target_msg


def _build_joint_target(np_action: np.ndarray, action_mode: ActionMode
) -> robot_service_pb2.JointTarget:
    """Creates a RobotDesired message from an action slice."""
    des_target_msg = robot_service_pb2.JointTarget()
    for i, joint_name in enumerate(CANONICAL_ARM_JOINTS):
        des_target_msg.joint_angles[joint_name] = np_action[i]

    # Action is always in absolute format after action_translator, so use absolute indices for grippers
    abs_mode = ActionMode.get_absolute_mode(action_mode)
    des_target_msg.gripper_widths["left"] = np_action[
        GET_LEFT_GRIPPER_IDX(abs_mode)
    ]
    des_target_msg.gripper_widths["right"] = np_action[
        GET_RIGHT_GRIPPER_IDX(abs_mode)
    ]

    des_target_msg.robot_stiffness_factors["left"] = 0.7
    des_target_msg.robot_stiffness_factors["right"] = 0.7

    return des_target_msg

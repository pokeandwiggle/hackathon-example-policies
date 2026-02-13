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

from ...utils.state_order import CANONICAL_ARM_JOINTS
from .robot_service import robot_service_pb2, robot_service_pb2_grpc


class RobotClient:
    CART_QUEUE = "cartesian_target_queue"
    CART_DIRECT = "cartesian_target"
    CART_WAYPOINT = "cartesian_waypoint"
    JOINT_DIRECT = "joint_target"

    def __init__(self, service_stub: robot_service_pb2_grpc.RobotServiceStub):
        self.stub = service_stub
        self.control_mode = None

    def get_snapshot(self):
        snapshot_request = robot_service_pb2.GetStateRequest()
        snapshot_response = self.stub.GetState(snapshot_request)

        state = snapshot_response.current_state

        if not state.robots:
            print("No robots found in snapshot")
            return None, None

        robot_names = list(state.robots.keys())

        return state, robot_names

    def send_cart_queue_target(self, cart_target: robot_service_pb2.CartesianTarget):
        ctrl_mode = RobotClient.CART_QUEUE

        if self.control_mode != ctrl_mode:
            prepare_request = robot_service_pb2.PrepareExecutionRequest()
            prepare_request.execution_mode = (
                robot_service_pb2.ExecutionMode.EXECUTION_MODE_CARTESIAN_TARGET_QUEUE
            )
            response = self.stub.PrepareExecution(prepare_request)
            self.control_mode = ctrl_mode

        queue_target_request = robot_service_pb2.EnqueueCartesianTargetsRequest()
        queue_target_request.cartesian_targets.append(cart_target)
        response = self.stub.EnqueueCartesianTargets(queue_target_request)
        return response

    def send_cart_waypoint(self, cart_target: robot_service_pb2.CartesianTarget):
        ctrl_mode = RobotClient.CART_WAYPOINT

        if self.control_mode != ctrl_mode:
            prepare_request = robot_service_pb2.PrepareExecutionRequest()
            prepare_request.execution_mode = (
                robot_service_pb2.ExecutionMode.EXECUTION_MODE_CARTESIAN_WAYPOINT
            )
            response = self.stub.PrepareExecution(prepare_request)
            self.control_mode = ctrl_mode

        waypoint_request = robot_service_pb2.SetCartesianWaypointRequest()
        waypoint_request.cartesian_waypoint.CopyFrom(cart_target)
        response = self.stub.SetCartesianWaypoint(waypoint_request)
        return response

    def send_cart_direct_target(self, cart_target: robot_service_pb2.CartesianTarget):
        ctrl_mode = RobotClient.CART_DIRECT

        if self.control_mode != ctrl_mode:
            prepare_request = robot_service_pb2.PrepareExecutionRequest()
            prepare_request.execution_mode = (
                robot_service_pb2.ExecutionMode.EXECUTION_MODE_CARTESIAN_TARGET
            )
            response = self.stub.PrepareExecution(prepare_request)
            self.control_mode = ctrl_mode

        set_target_request = robot_service_pb2.SetCartesianTargetRequest()
        set_target_request.cartesian_target.CopyFrom(cart_target)

        # Currently not safe
        response = self.stub.SetCartesianTarget(set_target_request)
        return response

    def send_joint_direct_target(self, joint_target: robot_service_pb2.JointTarget):
        ctrl_mode = RobotClient.JOINT_DIRECT

        if self.control_mode != ctrl_mode:
            prepare_request = robot_service_pb2.PrepareExecutionRequest()
            prepare_request.execution_mode = (
                robot_service_pb2.ExecutionMode.EXECUTION_MODE_JOINT_TARGET
            )
            response = self.stub.PrepareExecution(prepare_request)
            self.control_mode = ctrl_mode

        set_target_request = robot_service_pb2.SetJointTargetRequest()
        set_target_request.joint_target.CopyFrom(joint_target)

        # Currently not safe
        response = self.stub.SetJointTarget(set_target_request)
        return response

    def move_to_joint_goal(
        self, joint_angles: np.ndarray, joint_names: list[str] = CANONICAL_ARM_JOINTS
    ):
        """
        Moves the robot to a specific joint configuration using trajectory planning.

        Args:
            joint_angles: Array of target joint angles (radians)
            joint_names: List of joint names corresponding to the angles.
                        Defaults to CANONICAL_ARM_JOINTS (left + right arm joints)

        Returns:
            The response from the MoveToJointGoal gRPC call

        Raises:
            Exception: If the move fails, with the error message from the response
            ValueError: If joint_angles and joint_names have different lengths
        """
        # Validate input lengths match
        if len(joint_angles) != len(joint_names):
            raise ValueError(
                f"Length mismatch: joint_angles has {len(joint_angles)} elements "
                f"but joint_names has {len(joint_names)} elements"
            )

        # Reset control_mode because moving to a goal is a special operation
        self.control_mode = None

        # Build the request with joint angle map
        request = robot_service_pb2.MoveToJointGoalRequest()
        for name, angle in zip(joint_names, joint_angles):
            request.joint_angles[name] = float(angle)

        response = self.stub.MoveToJointGoal(request)

        if not response.success:
            raise Exception(f"Failed to move to joint goal: {response.error_message}")

        return response

    def set_gripper_state(
        self, gripper_id: str, width: float, speed: float = 0.0, force: float = 0.0
    ):
        """
        Controls the gripper state (open/close).

        Args:
            gripper_id: Gripper identifier ("left" or "right")
            width: Desired gripper width in meters
            speed: Gripper speed in m/s
            force: Gripper force in Newtons

        Returns:
            The response from the SetGripperState gRPC call
        """
        request = robot_service_pb2.SetGripperStateRequest()
        request.gripper_id = gripper_id
        request.width = width
        request.speed = speed
        request.force = force

        response = self.stub.SetGripperState(request)
        return response

    def visualize_cart_direct_targets(
        self, cart_targets: list[robot_service_pb2.CartesianTarget]
    ):
        """
        Visualizes a sequence of Cartesian targets for all arms as separate trajectories.

        Args:
            cart_targets: List of CartesianTarget messages to visualize

        Returns:
            Dictionary mapping arm_id to response
        """
        # Group poses by arm_id
        arm_requests = {}

        for cart_target in cart_targets:
            for arm_id, pose in cart_target.robot_poses.items():
                if arm_id not in arm_requests:
                    request = robot_service_pb2.VisualizeCartesianTargetActionChunkRequest()
                    request.arm_id = arm_id
                    arm_requests[arm_id] = request
                arm_requests[arm_id].poses.append(pose)

        # Send visualization requests for each arm
        responses = {}
        for arm_id, request in arm_requests.items():
            if request.poses:
                responses[arm_id] = self.stub.VisualizeCartesianTargetActionChunk(request)

        return responses

    def visualize_joint_direct_targets(
        self, joint_targets: list[robot_service_pb2.JointTarget], arm_id: str = "right"
    ):
        """
        Visualizes a sequence of joint targets.

        Args:
            joint_targets: List of JointTarget messages to visualize

        Returns:
            The response from the visualization gRPC call
        """
        request = robot_service_pb2.VisualizeJointConfigurationPathRequest()
        # Use "right" as default arm_id (can be extended to support both arms)
        request.arm_id = arm_id

        for joint_target in joint_targets:
            config = robot_service_pb2.JointConfiguration()
            config.joint_angles.update(joint_target.joint_angles)
            request.configurations.append(config)

        response = self.stub.VisualizeJointConfigurationPath(request)
        return response
    
    def send_move_home(self):
        """
        Sends a request to move the robot to its home position and resets the control mode.

        Returns:
            The response from the MoveHome gRPC call.
        """
        # Reset control_mode because moving home is a special operation that does not use the previous control mode.
        self.control_mode = None

        move_home_request = robot_service_pb2.MoveHomeRequest()
        response = self.stub.MoveHome(move_home_request)
        return response

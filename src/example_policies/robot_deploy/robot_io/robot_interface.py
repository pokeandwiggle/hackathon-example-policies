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

import cv2
import numpy as np
import torch

from example_policies import data_constants as dc

from ..debug_helpers import sensor_stream as dbg_sensors
from .observation_builder import ObservationBuilder
from .robot_client import RobotClient
from .robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


class RobotInterface:
    """Handles communication and data conversion with the robot gRPC service."""

    def __init__(self, service_stub: robot_service_pb2_grpc.RobotServiceStub, cfg):
        self.client = RobotClient(service_stub)
        self.observation_builder = ObservationBuilder(cfg)
        self.robot_names = None
        self.last_command = None

    def get_observation(self, device, show=False):
        """Gets the current observation from the robot."""
        snapshot_response, self.robot_names = self.client.get_snapshot()

        if show:
            dbg_sensors.show_response(snapshot_response)
            cv2.waitKey(1)

        obs = self.observation_builder.get_observation(
            snapshot_response, self.last_command, device
        )
        return obs

    def send_action(
        self,
        action: torch.Tensor,
    ):
        """Sends a predicted action to the robot service."""
        numpy_action = action.squeeze(0).to("cpu").numpy()

        self.last_command = numpy_action[: dc.LEFT_GRIPPER_IDX]

        left_desired = _build_des_msg(
            numpy_action[dc.LEFT_ARM], numpy_action[dc.LEFT_GRIPPER_IDX]
        )
        right_desired = _build_des_msg(
            numpy_action[dc.RIGHT_ARM], numpy_action[dc.RIGHT_GRIPPER_IDX]
        )

        target = robot_service_pb2.Target()
        target.robots["left"].CopyFrom(left_desired)
        target.robots["right"].CopyFrom(right_desired)

        response = self.client.send_target(target)
        print(response)


def _build_des_msg(
    action_slice: np.ndarray, gripper_width: float
) -> robot_service_pb2.RobotDesired:
    """Creates a RobotDesired message from an action slice."""
    desired_msg = robot_service_pb2.RobotDesired()
    desired_msg.pose.position.x = action_slice[0]
    desired_msg.pose.position.y = action_slice[1]
    desired_msg.pose.position.z = action_slice[2]
    desired_msg.pose.orientation.x = action_slice[3]
    desired_msg.pose.orientation.y = action_slice[4]
    desired_msg.pose.orientation.z = action_slice[5]
    desired_msg.pose.orientation.w = action_slice[6]

    desired_msg.gripper_width = gripper_width
    return desired_msg

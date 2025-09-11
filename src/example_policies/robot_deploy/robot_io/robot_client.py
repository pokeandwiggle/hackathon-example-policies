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

from .robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


class RobotClient:
    def __init__(self, service_stub: robot_service_pb2_grpc.RobotServiceStub):
        self.stub = service_stub

    def get_snapshot(self):
        snapshot_request = robot_service_pb2.GetSnapshotRequest()
        snapshot_response = self.stub.GetSnapshot(snapshot_request)

        if not snapshot_response.robots:
            print("No robots found in snapshot")
            return None, None

        robot_names = list(snapshot_response.robots.keys())

        return snapshot_response, robot_names

    def send_target(self, target: robot_service_pb2.Target):
        set_target_request = robot_service_pb2.SetTargetRequest()
        set_target_request.targets.append(target)
        response = self.stub.SetTarget(set_target_request)
        return response

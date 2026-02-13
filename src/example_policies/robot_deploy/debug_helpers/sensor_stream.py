#!/usr/bin/env python

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

import argparse
import time

import cv2
import grpc

from example_policies.data_ops.utils import image_processor
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


def show_response(snapshot_response):
    images = {}
    cameras = snapshot_response.cameras
    # ['cam_right_color_optical_frame', 'cam_right_depth_optical_frame', 'cam_static_color_optical_frame', 'cam_left_depth_optical_frame', 'cam_left_color_optical_frame']
    for cam_name in list(cameras.keys()):
        images.update(process_camera_image(cameras[cam_name], cam_name, "cpu"))

    for cam_name, img in images.items():
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imshow(cam_name, img)


def process_camera_image(camera_data, camera_name, device):
    side = camera_name.split("_")[1]
    modality = "depth" if "depth" in camera_name else "rgb"

    obs_key = f"observation.images.{modality}_{side}"

    img_array = image_processor.process_image_bytes(
        camera_data.data,
        width=640,
        height=480,
        is_depth=(modality == "depth"),
    )
    return {obs_key: img_array}


def main(service_stub: robot_service_pb2_grpc.RobotServiceStub):
    """Connects to the robot service and displays the camera streams."""
    print("Starting debug stream. Press 'q' in any window to quit.")

    while True:
        try:
            # Request a snapshot of the robot's current state
            snapshot_request = robot_service_pb2.GetStateRequest()
            snapshot_response = service_stub.GetState(snapshot_request)

            show_response(snapshot_response.current_state)

            # Wait for a key press. If 'q' is pressed, exit the loop.
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

            # Small delay to prevent overwhelming the service
            time.sleep(0.01)

        except grpc.RpcError as e:
            print(f"An RPC error occurred: {e.code()} - {e.details()}")
            print("Attempting to reconnect in 5 seconds...")
            time.sleep(5)
        except Exception as e:
            print(f"An unexpected error occurred: {e}")
            raise e
            break

    cv2.destroyAllWindows()
    print("Debug stream stopped.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Robot stream visualization client")
    parser.add_argument(
        "-s",
        "--server",
        default="localhost:50051",
        metavar="ADDR",
        help="Robot service server address (default: localhost:50051)",
    )
    args = parser.parse_args()

    channel = grpc.insecure_channel(args.server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)

    try:
        main(stub)
    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        channel.close()
        print("Connection closed.")

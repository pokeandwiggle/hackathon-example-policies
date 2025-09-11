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
import io
import time

import cv2
import grpc
import numpy as np
from PIL import Image

from example_policies.data_ops.utils import image_processor
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


def convert_image_for_display(img_bytes: bytes, is_depth: bool) -> np.ndarray:
    """Converts image bytes from the service to a displayable OpenCV format (BGR, uint8)."""
    # Open image from bytes using Pillow
    # pil_img = Image.open(io.BytesIO(img_bytes))
    # img_array = np.array(pil_img)

    img_array = image_processor.process_image_bytes(img_bytes, 640, 640, is_depth)

    display_image = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    return display_image


def show_response(snapshot_response):
    # Process and display RGB images
    for cam_name, cam_data in snapshot_response.rgb_cameras.items():
        if cam_data.data:
            rgb_image = convert_image_for_display(cam_data.data, is_depth=False)
            cv2.imshow(f"RGB - {cam_name}", rgb_image)

    # Process and display depth images
    for cam_name, cam_data in snapshot_response.depth_cameras.items():
        if cam_data.data:
            depth_image = convert_image_for_display(cam_data.data, is_depth=True)
            cv2.imshow(f"Depth - {cam_name}", depth_image)


def main(service_stub: robot_service_pb2_grpc.RobotServiceStub):
    """Connects to the robot service and displays the camera streams."""
    print("Starting debug stream. Press 'q' in any window to quit.")

    while True:
        try:
            # Request a snapshot of the robot's current state
            snapshot_request = robot_service_pb2.GetSnapshotRequest()
            snapshot_response = service_stub.GetSnapshot(snapshot_request)

            show_response(snapshot_response)

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
        "--server",
        default="localhost:50051",
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

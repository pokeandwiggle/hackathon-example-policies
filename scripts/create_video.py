#!/usr/bin/env python3
"""
Simple script to create an MP4 video from RGB_LEFT_IMAGE topic in a .mcap file.

Usage:
    python scripts/create_video.py path/to/file.mcap
"""

import argparse
import pathlib

import cv2
import numpy as np
from mcap.reader import NonSeekingReader
from rosbags.serde import deserialize_cdr

from example_policies.data_ops.config.rosbag_topics import RosTopicEnum


def decode_compressed_image(msg_data, schema_name):
    """Decode a ROS sensor_msgs/CompressedImage message."""
    # Use proper ROS deserialization
    img_msg = deserialize_cdr(msg_data, schema_name)

    # Extract timestamp from header
    timestamp = img_msg.header.stamp.sec + img_msg.header.stamp.nanosec / 1e9

    # Get image data
    img_bytes = img_msg.data
    is_depth = "compressedDepth" in img_msg.format

    # For depth images, skip the first 12 bytes (PNG header modifications)
    png_data = img_bytes[12:] if is_depth else img_bytes

    # Decode the compressed image
    np_arr = np.frombuffer(png_data, np.uint8)
    read_flag = cv2.IMREAD_UNCHANGED if is_depth else cv2.IMREAD_COLOR
    image = cv2.imdecode(np_arr, read_flag)

    # For RGB images, convert from BGR to RGB
    if not is_depth and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    return image, timestamp


def create_video_from_mcap(mcap_path, output_path=None):
    """Extract images from RGB_LEFT_IMAGE topic and create MP4 video."""
    mcap_path = pathlib.Path(mcap_path)
    if output_path is None:
        output_path = mcap_path.parent / f"{mcap_path.stem}_rgb_left.mp4"

    target_topic = (
        RosTopicEnum.RGB_LEFT_IMAGE.value
    )  # "/cam_left/color/image_rect_compressed"

    images = []
    timestamps = []

    print(f"Reading {mcap_path}...")
    print(f"Looking for topic: {target_topic}")

    with open(mcap_path, "rb") as f:
        reader = NonSeekingReader(f, record_size_limit=None)

        for schema, channel, message in reader.iter_messages():
            if channel.topic == target_topic:
                try:
                    image, timestamp = decode_compressed_image(
                        message.data, schema.name
                    )
                    if image is not None:
                        images.append(image)
                        timestamps.append(timestamp)
                        if len(images) % 10 == 0:
                            print(f"Extracted {len(images)} images...", end="\r")
                except Exception as e:
                    print(f"Failed to decode image: {e}")
                    continue

    if not images:
        print(f"No images found for topic {target_topic}")
        return

    print(f"\nExtracted {len(images)} images")

    # Calculate FPS from timestamps
    if len(timestamps) > 1:
        time_diffs = np.diff(timestamps)
        avg_dt = np.mean(time_diffs)
        fps = 1.0 / avg_dt if avg_dt > 0 else 30.0
    else:
        fps = 30.0

    fps = min(fps, 60.0)  # Cap at 60 FPS
    print(f"Estimated FPS: {fps:.2f}")

    # Create video
    if images:
        height, width, channels = images[0].shape
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

        print(f"Creating video {output_path}...")
        for i, image in enumerate(images):
            # Convert RGB back to BGR for OpenCV video writer
            bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            video_writer.write(bgr_image)
            if i % 10 == 0:
                print(f"Writing frame {i + 1}/{len(images)}...", end="\r")

        video_writer.release()
        print(f"\nVideo saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Create MP4 video from RGB_LEFT_IMAGE topic in .mcap file"
    )
    parser.add_argument("mcap_file", type=pathlib.Path, help="Path to the .mcap file")
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        help="Output video path (default: same directory as input with _rgb_left.mp4 suffix)",
    )

    args = parser.parse_args()

    if not args.mcap_file.exists():
        raise FileNotFoundError(f"Input file not found: {args.mcap_file}")

    create_video_from_mcap(args.mcap_file, args.output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3

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

"""
Episode Video Reviewer for LeRobot Dataset using torchcodec/PyAV with OpenCV display

Usage: python review_dataset.py [--start-episode N]

Controls (when video windows are focused):
- 'b' or 'B': Add current episode to blacklist and go to next
- 'n' or 'N': Whitelist/skip to next episode (removes from blacklist if present)
- Space: Pause/resume current videos
- 'r' or 'R': Replay current video from beginning
- Left/Right Arrow: Step backward/forward one frame (auto-pauses)
- 'q' or 'Q' or Escape: Quit
- Any other key: Pause/resume current videos
"""

import argparse
import json
import sys
from importlib.util import find_spec
from pathlib import Path

import cv2
import numpy as np

from example_policies.utils.constants import BLACKLIST_FILE, META_DIR, VIDEO_DIR

VIDEO_BACKEND = None  # Will be set to 'torchcodec', 'pyav', or 'opencv'

# Window display constants
WINDOW_WIDTH = 640
WINDOW_HEIGHT = 480
WINDOW_HORIZONTAL_OFFSET = 660
WINDOW_VERTICAL_OFFSET = 50

# Key codes (may vary by system)
# The arrow key constants are tuples to handle platform differences:
#   - The first value is the key code on Linux (X11/GTK/Qt, etc.)
#   - The second value is the key code on macOS (Cocoa/Quartz, etc.)
# If you encounter issues, check your system's key codes with OpenCV's waitKey.
KEY_ESCAPE = 27
KEY_LEFT_ARROW = (81, 2)
KEY_RIGHT_ARROW = (83, 3)
KEY_NO_INPUT = 255


def setup_video_backend():
    """Setup the best available video backend"""
    global VIDEO_BACKEND

    # Try torchcodec first (best for ML datasets)
    if find_spec("torchcodec") is not None:
        VIDEO_BACKEND = "torchcodec"
        print("Using torchcodec for video decoding")
        return True

    # Try PyAV (excellent codec support)
    if find_spec("av") is not None:
        VIDEO_BACKEND = "pyav"
        print("Using PyAV for video decoding")
        return True

    # Fallback to OpenCV
    VIDEO_BACKEND = "opencv"
    print("Using OpenCV for video decoding (may have codec issues)")
    return True


class VideoReader:
    """Unified video reader interface"""

    def __init__(self, video_path, frames_to_load=10):
        self.video_path = str(video_path)
        self.frames = []
        self.frame_indices = []  # Track which frame indices we loaded
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30  # Default FPS
        self.frames_to_load = frames_to_load  # Number of frames from start/end

        if VIDEO_BACKEND == "torchcodec":
            self._init_torchcodec()
        elif VIDEO_BACKEND == "pyav":
            self._init_pyav()
        else:
            self._init_opencv()

    def _determine_frames_to_load(self):
        """Determine which frames to load based on total_frames.

        Returns:
            tuple: (frame_ranges, frame_indices) where:
                - frame_ranges: list of (start, end) tuples for loading
                - frame_indices: list of frame numbers in the original video
        """
        if self.total_frames <= self.frames_to_load * 2:
            # Video is short, load all frames
            frame_ranges = [(0, self.total_frames)]
            frame_indices = list(range(self.total_frames))
        else:
            # Load first N and last N frames
            frame_ranges = [
                (0, self.frames_to_load),
                (self.total_frames - self.frames_to_load, self.total_frames),
            ]
            frame_indices = list(range(self.frames_to_load)) + list(
                range(self.total_frames - self.frames_to_load, self.total_frames)
            )

        return frame_ranges, frame_indices

    def _select_frames_from_list(self, all_frames):
        """Select frames from a list of all loaded frames.

        Args:
            all_frames: List of all frames loaded from video

        Returns:
            List of selected frames (all frames if short video, or first N + last N)
        """
        _, self.frame_indices = self._determine_frames_to_load()

        if self.total_frames <= self.frames_to_load * 2:
            # Video is short, keep all frames
            return all_frames
        else:
            # Keep only first N and last N frames
            return (
                all_frames[: self.frames_to_load] + all_frames[-self.frames_to_load :]
            )

    def _init_torchcodec(self):
        """Initialize with torchcodec - only load first N and last N frames"""
        try:
            from torchcodec.decoders import VideoDecoder

            decoder = VideoDecoder(self.video_path)

            # Get video info
            self.fps = (
                float(decoder.metadata.average_fps)
                if hasattr(decoder.metadata, "average_fps")
                else 30.0
            )
            self.total_frames = (
                int(decoder.metadata.num_frames)
                if hasattr(decoder.metadata, "num_frames")
                else 1000
            )

            # Determine which frames to load
            frame_ranges, self.frame_indices = self._determine_frames_to_load()

            # Decode selected frame ranges
            all_frames = []
            for start, end in frame_ranges:
                frames_batch = decoder.get_frames_in_range(start, end)

                # Handle different torchcodec return types
                if hasattr(frames_batch, "data"):
                    frames_tensor = frames_batch.data
                elif hasattr(frames_batch, "permute"):
                    frames_tensor = frames_batch
                else:
                    frames_tensor = frames_batch

                # Convert to numpy arrays
                if hasattr(frames_tensor, "permute"):
                    frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
                elif hasattr(frames_tensor, "transpose"):
                    if len(frames_tensor.shape) == 4 and frames_tensor.shape[1] == 3:
                        frames_np = np.transpose(frames_tensor, (0, 2, 3, 1))
                    else:
                        frames_np = frames_tensor
                else:
                    frames_np = np.array(frames_tensor)

                # Ensure correct data type and range
                if frames_np.dtype == np.float32:
                    if frames_np.max() <= 1.0:
                        frames_np = (frames_np * 255).astype(np.uint8)
                    else:
                        frames_np = frames_np.astype(np.uint8)
                elif frames_np.dtype != np.uint8:
                    frames_np = frames_np.astype(np.uint8)

                # Ensure BGR format for OpenCV (torchcodec often returns RGB)
                if len(frames_np.shape) == 4 and frames_np.shape[-1] == 3:
                    frames_np = frames_np[:, :, :, [2, 1, 0]]

                all_frames.extend([frame for frame in frames_np])

            self.frames = all_frames
            self.success = True

        except Exception as e:
            print(f"    ✗ Torchcodec failed for {self.video_path}: {e}")
            self.success = False

    def _init_pyav(self):
        """Initialize with PyAV - only load first N and last N frames"""
        try:
            import av

            container = av.open(self.video_path)
            video_stream = container.streams.video[0]

            self.fps = float(video_stream.average_rate)
            self.total_frames = video_stream.frames

            # First pass: decode all frames to count them (PyAV frame count can be unreliable)
            all_frames_temp = []
            for frame in container.decode(video_stream):
                img = frame.to_ndarray(format="bgr24")
                all_frames_temp.append(img)

            self.total_frames = len(all_frames_temp)

            # Select frames to keep
            self.frames = self._select_frames_from_list(all_frames_temp)

            self.success = True
            container.close()

        except Exception as e:
            print(f"    ✗ PyAV failed for {self.video_path}: {e}")
            self.success = False

    def _init_opencv(self):
        """Initialize with OpenCV (fallback) - only load first N and last N frames"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.success = False
                return

            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            # Read all frames first (OpenCV seeking can be unreliable)
            all_frames_temp = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                all_frames_temp.append(frame)

            self.total_frames = len(all_frames_temp)

            # Select frames to keep
            self.frames = self._select_frames_from_list(all_frames_temp)

            self.success = True
            cap.release()

        except Exception as e:
            print(f"    ✗ OpenCV failed for {self.video_path}: {e}")
            self.success = False

    def get_frame(self, frame_idx):
        """Get frame by index (into our loaded frames array)"""
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None

    def get_actual_frame_number(self, frame_idx):
        """Get the actual frame number in the original video"""
        if 0 <= frame_idx < len(self.frame_indices):
            return self.frame_indices[frame_idx]
        return frame_idx

    def __len__(self):
        return len(self.frames)


class EpisodeReviewer:
    @staticmethod
    def clean_video_dir_name(video_dir):
        """Remove prefix from video directory name for display"""
        return video_dir.replace("observation.images.", "")

    @staticmethod
    def create_frame_info_string(reader, current_frame):
        """Create frame information string for overlay display"""
        actual_frame_num = reader.get_actual_frame_number(current_frame)
        frame_num_str = f"Frame {actual_frame_num + 1}/{reader.total_frames}"
        loaded_str = f"Loaded: {len(reader)}/{reader.total_frames}"
        backend_str = f"Backend: {VIDEO_BACKEND}"
        return f"{frame_num_str} | {loaded_str} | {backend_str}"

    def __init__(self, data_root=".", start_episode=0):
        self.data_root = Path(data_root)
        self.current_episode = start_episode
        self.blacklist_path = self.data_root / META_DIR / BLACKLIST_FILE
        self.video_base = self.data_root / "videos" / VIDEO_DIR.split("/")[-1]

        # Video subdirectories
        self.video_dirs = [
            "observation.images.rgb_left",
            "observation.images.rgb_right",
            "observation.images.rgb_static",
        ]

        # Load existing blacklist
        self.blacklist = self.load_blacklist()

        # Find max episode number
        self.max_episode = self.find_max_episode()

        print(f"Found episodes 0-{self.max_episode}")
        print(f"Starting from episode {self.current_episode}")
        print(f"Video backend: {VIDEO_BACKEND}")
        print("\nControls (when video windows are focused):")
        print("- 'b': Add to blacklist and continue")
        print("- 'n': Whitelist/skip to next (removes from blacklist if present)")
        print("- Space: Pause/resume")
        print("- 'r': Replay (restart from beginning)")
        print("- Left/Right Arrow: Step backward/forward one frame")
        print("- 'q' or Escape: Quit")
        print("- Any other key: Pause/resume")
        print("-" * 50)

    def load_blacklist(self):
        """Load existing blacklist or create empty one"""
        try:
            if self.blacklist_path.exists():
                with open(self.blacklist_path, "r") as f:
                    return json.load(f)
            else:
                return []
        except (json.JSONDecodeError, FileNotFoundError):
            print(f"Warning: Could not load blacklist from {self.blacklist_path}")
            return []

    def save_blacklist(self):
        """Save blacklist to JSON file"""
        try:
            # Ensure meta directory exists
            self.blacklist_path.parent.mkdir(parents=True, exist_ok=True)

            with open(self.blacklist_path, "w") as f:
                json.dump(sorted(list(set(self.blacklist))), f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving blacklist: {e}")
            return False

    def find_max_episode(self):
        """Find the highest episode number"""
        max_ep = -1
        for video_dir in self.video_dirs:
            dir_path = self.video_base / video_dir
            if dir_path.exists():
                for video_file in dir_path.glob("episode_*.mp4"):
                    try:
                        ep_num = int(video_file.stem.split("_")[1])
                        max_ep = max(max_ep, ep_num)
                    except (ValueError, IndexError):
                        continue
        return max_ep

    def get_episode_videos(self, episode_num):
        """Get all video files for an episode"""
        videos = []
        episode_name = f"episode_{episode_num:06d}.mp4"

        for video_dir in self.video_dirs:
            video_path = self.video_base / video_dir / episode_name
            if video_path.exists():
                videos.append((video_path, video_dir))

        return videos

    def create_status_overlay(
        self,
        frame,
        episode_num,
        video_name,
        blacklisted=False,
        frame_info="",
        paused=False,
    ):
        """Add status overlay to frame"""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (min(w - 10, 650), 145), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Text
        status = "BLACKLISTED" if blacklisted else "REVIEWING"
        if paused:
            status += " (PAUSED)"
        color = (0, 0, 255) if blacklisted else (0, 255, 0)

        cv2.putText(
            frame,
            f"Episode: {episode_num:06d} ({status})",
            (20, 35),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            color,
            2,
        )
        cv2.putText(
            frame,
            f"Camera: {self.clean_video_dir_name(video_name)}",
            (20, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
        )
        cv2.putText(
            frame,
            frame_info,
            (20, 80),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "b=Blacklist | n=Whitelist/Next | r=Replay | <-/->=Step",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            frame,
            "Space=Pause | q=Quit",
            (20, 130),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (200, 200, 200),
            1,
        )

        return frame

    def play_videos_sync(self, video_paths, episode_num):
        """Play multiple videos synchronously"""
        readers = []
        window_names = []

        # Load all videos
        print(f"  Loading {len(video_paths)} videos...")
        for video_path, video_dir in video_paths:
            clean_name = self.clean_video_dir_name(video_dir)
            print(
                f"    Loading {clean_name}...",
                end=" ",
            )
            reader = VideoReader(video_path)

            if reader.success and len(reader) > 0:
                readers.append((reader, video_dir))
                window_name = f"Episode {episode_num:06d} - {clean_name}"
                window_names.append(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, WINDOW_WIDTH, WINDOW_HEIGHT)
                print(f"✓ ({len(reader)} frames, {reader.fps:.1f} FPS)")
            else:
                print("✗")

        if not readers:
            print("    No videos could be loaded")
            return "n"

        # Position windows
        for i, window_name in enumerate(window_names):
            cv2.moveWindow(
                window_name, i * WINDOW_HORIZONTAL_OFFSET, WINDOW_VERTICAL_OFFSET
            )

        # Find minimum frame count for synchronization
        min_frames = min(len(reader) for reader, _ in readers)
        max_fps = max(reader.fps for reader, _ in readers)
        frame_delay = max(1, int(1000 / max_fps))  # milliseconds

        print(
            f"  Playing {len(readers)} videos ({min_frames} frames, {frame_delay}ms delay)"
        )

        paused = False
        blacklisted = episode_num in self.blacklist
        current_frame = 0
        playing = True  # Track if we're in playback mode

        def display_current_frame():
            """Helper to display current frame from all videos"""
            for i, (reader, video_dir) in enumerate(readers):
                frame = reader.get_frame(current_frame)
                if frame is not None:
                    frame_info = self.create_frame_info_string(reader, current_frame)
                    frame_with_overlay = self.create_status_overlay(
                        frame, episode_num, video_dir, blacklisted, frame_info, paused
                    )
                    cv2.imshow(window_names[i], frame_with_overlay)

        try:
            while playing:
                # Display current frame
                if not paused:
                    display_current_frame()
                    current_frame += 1

                    # Check if we've reached the end
                    if current_frame >= min_frames:
                        current_frame = min_frames - 1  # Stay at last frame
                        paused = True
                        print(
                            "    Video finished. Press 'r' to replay, or 'n' to continue..."
                        )
                else:
                    # When paused, still display the current frame
                    display_current_frame()

                # Handle keyboard input
                key = cv2.waitKey(frame_delay if not paused else 10) & 0xFF

                if key == ord("q") or key == KEY_ESCAPE:
                    return "q"
                elif key == ord("b") or key == ord("B"):
                    return "b"
                elif key == ord("n") or key == ord("N"):
                    return "n"
                elif key == ord(" "):  # Space bar pauses/resumes
                    paused = not paused
                    status = "PAUSED" if paused else "PLAYING"
                    print(f"    {status} (frame {current_frame + 1}/{min_frames})")
                elif key == ord("r") or key == ord("R"):
                    current_frame = 0
                    paused = False
                    print("    Replaying from beginning...")
                elif key in KEY_LEFT_ARROW:
                    if current_frame > 0:
                        current_frame -= 1
                    paused = True
                    print(f"    Frame {current_frame + 1}/{min_frames}")
                elif key in KEY_RIGHT_ARROW:
                    if current_frame < min_frames - 1:
                        current_frame += 1
                    paused = True
                    print(f"    Frame {current_frame + 1}/{min_frames}")
                elif (
                    key != KEY_NO_INPUT
                    and key not in KEY_LEFT_ARROW
                    and key not in KEY_RIGHT_ARROW
                ):
                    paused = not paused
                    status = "PAUSED" if paused else "PLAYING"
                    print(f"    {status} (frame {current_frame + 1}/{min_frames})")

        except KeyboardInterrupt:
            return "q"

        finally:
            # Clean up windows
            for window_name in window_names:
                cv2.destroyWindow(window_name)
            cv2.waitKey(1)

    def review_episode(self, episode_num):
        """Review a single episode"""
        if episode_num in self.blacklist:
            status = " [BLACKLISTED]"
        else:
            status = ""

        print(f"\nEpisode {episode_num:06d}{status}")

        # Get videos for this episode
        videos = self.get_episode_videos(episode_num)

        if not videos:
            print(f"  No videos found for episode {episode_num}")
            return "n"  # Skip to next

        return self.play_videos_sync(videos, episode_num)

    def run(self):
        """Main review loop"""
        try:
            while self.current_episode <= self.max_episode:
                action = self.review_episode(self.current_episode)

                if action == "q":
                    break
                elif action == "b":
                    if self.current_episode not in self.blacklist:
                        self.blacklist.append(self.current_episode)
                        if self.save_blacklist():
                            print(
                                f"  ✓ Episode {self.current_episode} added to blacklist"
                            )
                        else:
                            print("  ✗ Failed to save blacklist")
                    else:
                        print(f"  Episode {self.current_episode} already in blacklist")
                    self.current_episode += 1
                elif action == "n":
                    if self.current_episode in self.blacklist:
                        self.blacklist.remove(self.current_episode)
                        if self.save_blacklist():
                            print(
                                f"  ✓ Episode {self.current_episode} removed from blacklist"
                            )
                        else:
                            print("  ✗ Failed to save blacklist")
                    self.current_episode += 1

            print(f"\nReview complete! Blacklisted episodes: {len(self.blacklist)}")
            if self.blacklist:
                print(f"Blacklisted: {sorted(self.blacklist)}")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            cv2.destroyAllWindows()
            self.save_blacklist()


def main():
    # Setup video backend first
    if not setup_video_backend():
        print("Error: No video backend available")
        sys.exit(1)

    parser = argparse.ArgumentParser(description="Review LeRobot dataset episodes")
    parser.add_argument(
        "--start-episode",
        "-s",
        type=int,
        default=0,
        help="Episode number to start from (default: 0)",
    )
    parser.add_argument(
        "--data-root",
        "-d",
        type=str,
        default=".",
        help="Root directory of the dataset (default: current directory)",
    )

    args = parser.parse_args()

    # Check if data structure exists
    data_root = Path(args.data_root)
    if not (data_root / "videos" / "chunk-000").exists():
        print(
            f"Error: Video directory not found at {data_root / 'videos' / 'chunk-000'}"
        )
        print("Make sure you're in the correct dataset directory")
        sys.exit(1)

    reviewer = EpisodeReviewer(data_root, args.start_episode)
    reviewer.run()


if __name__ == "__main__":
    main()

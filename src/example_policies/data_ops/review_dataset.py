#!/usr/bin/env python3
"""
Episode Video Reviewer for LeRobot Dataset using torchcodec/PyAV with OpenCV display

Usage: python review_dataset.py [--start-episode N]

Controls (when video windows are focused):
- 'b' or 'B': Add current episode to blacklist and go to next
- 'n' or 'N' or Space: Skip to next episode
- 'q' or 'Q' or Escape: Quit
- Any other key: Pause/resume current videos
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Try different video decoding backends
VIDEO_BACKEND = None


def setup_video_backend():
    """Setup the best available video backend"""
    global VIDEO_BACKEND

    # Try torchcodec first (best for ML datasets)
    try:
        import torchcodec
        from torchcodec.decoders import VideoDecoder

        VIDEO_BACKEND = "torchcodec"
        print("Using torchcodec for video decoding")
        return True
    except ImportError:
        pass

    # Try PyAV (excellent codec support)
    try:
        import av

        VIDEO_BACKEND = "pyav"
        print("Using PyAV for video decoding")
        return True
    except ImportError:
        pass

    # Fallback to OpenCV
    VIDEO_BACKEND = "opencv"
    print("Using OpenCV for video decoding (may have codec issues)")
    return True


class VideoReader:
    """Unified video reader interface"""

    def __init__(self, video_path):
        self.video_path = str(video_path)
        self.frames = []
        self.current_frame = 0
        self.total_frames = 0
        self.fps = 30  # Default FPS

        if VIDEO_BACKEND == "torchcodec":
            self._init_torchcodec()
        elif VIDEO_BACKEND == "pyav":
            self._init_pyav()
        else:
            self._init_opencv()

    def _init_torchcodec(self):
        """Initialize with torchcodec"""
        try:
            import torch
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

            # Decode all frames at once (for small videos this is fine)
            frames_batch = decoder.get_frames_in_range(0, self.total_frames)

            # Handle different torchcodec return types
            if hasattr(frames_batch, "data"):
                # FrameBatch object - extract tensor
                frames_tensor = frames_batch.data
            elif hasattr(frames_batch, "permute"):
                # Direct tensor
                frames_tensor = frames_batch
            else:
                # Try to access as tensor directly
                frames_tensor = frames_batch

            # Convert to numpy arrays
            if hasattr(frames_tensor, "permute"):
                # TCHW -> THWC format
                frames_np = frames_tensor.permute(0, 2, 3, 1).numpy()
            elif hasattr(frames_tensor, "transpose"):
                # Already numpy, transpose if needed
                if len(frames_tensor.shape) == 4 and frames_tensor.shape[1] == 3:
                    # NCHW -> NHWC
                    frames_np = np.transpose(frames_tensor, (0, 2, 3, 1))
                else:
                    frames_np = frames_tensor
            else:
                # Fallback - assume it's already in the right format
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
                # Convert RGB to BGR for OpenCV
                frames_np = frames_np[:, :, :, [2, 1, 0]]

            self.frames = [frame for frame in frames_np]
            self.total_frames = len(self.frames)
            self.success = True

        except Exception as e:
            print(f"    ✗ Torchcodec failed for {self.video_path}: {e}")
            self.success = False

    def _init_pyav(self):
        """Initialize with PyAV"""
        try:
            import av

            container = av.open(self.video_path)
            video_stream = container.streams.video[0]

            self.fps = float(video_stream.average_rate)
            self.total_frames = video_stream.frames

            frames = []
            for frame in container.decode(video_stream):
                # Convert to numpy array
                img = frame.to_ndarray(format="bgr24")  # OpenCV format
                frames.append(img)

            self.frames = frames
            self.total_frames = len(frames)
            self.success = True
            container.close()

        except Exception as e:
            print(f"    ✗ PyAV failed for {self.video_path}: {e}")
            self.success = False

    def _init_opencv(self):
        """Initialize with OpenCV (fallback)"""
        try:
            cap = cv2.VideoCapture(self.video_path)
            if not cap.isOpened():
                self.success = False
                return

            self.fps = cap.get(cv2.CAP_PROP_FPS)
            self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(frame)

            self.frames = frames
            self.total_frames = len(frames)
            self.success = True
            cap.release()

        except Exception as e:
            print(f"    ✗ OpenCV failed for {self.video_path}: {e}")
            self.success = False

    def get_frame(self, frame_idx):
        """Get frame by index"""
        if 0 <= frame_idx < len(self.frames):
            return self.frames[frame_idx]
        return None

    def __len__(self):
        return len(self.frames)


class EpisodeReviewer:
    def __init__(self, data_root=".", start_episode=0):
        self.data_root = Path(data_root)
        self.current_episode = start_episode
        self.blacklist_path = self.data_root / "meta" / "blacklist.json"
        self.video_base = self.data_root / "videos" / "chunk-000"

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
        print("- 'n' or Space: Next episode")
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
        self, frame, episode_num, video_name, blacklisted=False, frame_info=""
    ):
        """Add status overlay to frame"""
        if frame is None:
            return None

        h, w = frame.shape[:2]
        overlay = frame.copy()

        # Semi-transparent background for text
        cv2.rectangle(overlay, (10, 10), (min(w - 10, 600), 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(frame, 0.7, overlay, 0.3, 0)

        # Text
        status = "BLACKLISTED" if blacklisted else "REVIEWING"
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
            f"Camera: {video_name.replace('observation.images.', '')}",
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
            "b=Blacklist | n=Next | q=Quit | other=Pause",
            (20, 105),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
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
            print(
                f"    Loading {video_dir.replace('observation.images.', '')}...",
                end=" ",
            )
            reader = VideoReader(video_path)

            if reader.success and len(reader) > 0:
                readers.append((reader, video_dir))
                window_name = f"Episode {episode_num:06d} - {video_dir.replace('observation.images.', '')}"
                window_names.append(window_name)
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 640, 480)
                print(f"✓ ({len(reader)} frames, {reader.fps:.1f} FPS)")
            else:
                print("✗")

        if not readers:
            print("    No videos could be loaded")
            return "n"

        # Position windows
        for i, window_name in enumerate(window_names):
            cv2.moveWindow(window_name, i * 660, 50)

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

        try:
            while current_frame < min_frames:
                if not paused:
                    # Display current frame from all videos
                    for i, (reader, video_dir) in enumerate(readers):
                        frame = reader.get_frame(current_frame)
                        if frame is not None:
                            frame_info = f"Frame {current_frame+1}/{min_frames} | Backend: {VIDEO_BACKEND}"
                            frame_with_overlay = self.create_status_overlay(
                                frame, episode_num, video_dir, blacklisted, frame_info
                            )
                            cv2.imshow(window_names[i], frame_with_overlay)

                    current_frame += 1

                # Handle keyboard input
                key = cv2.waitKey(frame_delay if not paused else 0) & 0xFF

                if key == ord("q") or key == 27:  # 'q' or Escape
                    return "q"
                elif key == ord("b"):
                    return "b"
                elif key == ord("n") or key == ord(" "):  # 'n' or Space
                    return "n"
                elif key != 255:  # Any other key
                    paused = not paused
                    status = "PAUSED" if paused else "PLAYING"
                    print(f"    {status} (frame {current_frame}/{min_frames})")

            # End of video - wait for user input
            print(f"    Videos finished. Press any key to continue...")
            key = cv2.waitKey(0) & 0xFF
            if key == ord("q") or key == 27:
                return "q"
            elif key == ord("b"):
                return "b"
            else:
                return "n"

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
                            print(f"  ✗ Failed to save blacklist")
                    else:
                        print(f"  Episode {self.current_episode} already in blacklist")
                    self.current_episode += 1
                elif action == "n":
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

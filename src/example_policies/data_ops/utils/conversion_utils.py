"""Shared utility functions for dataset conversion."""

import json
import pathlib
import shutil
from mcap.reader import make_reader

from example_policies.utils.constants import (
    BLACKLIST_FILE,
    EPISODE_MAPPING_FILE,
    META_DIR,
    PIPELINE_CONFIG_FILE,
)


def get_sorted_episodes(episode_dir: pathlib.Path) -> list[pathlib.Path]:
    """Get episode paths sorted by creation time (oldest first).

    Args:
        episode_dir: Directory containing .mcap episode files

    Returns:
        List of episode paths sorted by creation date
    """
    episode_paths = list(episode_dir.rglob("*.mcap"))
    episode_paths.sort(key=lambda p: p.stat().st_ctime)
    return episode_paths

def get_selected_episodes(episode_dir: pathlib.Path, success_only=True):
    """Get episode paths sorted by creation time (oldest first), that fulfil success criteria in the MCAP file.

    Args:
        episode_dir: Directory containing .mcap episode files
        success_only: If True, include only successful episodes

    Returns:
        List of episode paths sorted by creation date that fulfil success criteria in the MCAP file
    """

    episode_paths = list(episode_dir.rglob("*.mcap"))
    episode_paths.sort(key=lambda p: p.stat().st_ctime)
    filtered_episode_paths = []

    if success_only is True:
        for ep_path in episode_paths:
            try:
                with open(ep_path, 'rb') as f:
                    reader = make_reader(f)

                    # Iterate through metadata records
                    for metadata_record in reader.iter_metadata():
                        if metadata_record.name == "recording_info":
                            # metadata is already a dict, no need to decode
                            metadata = metadata_record.metadata
                            quality = metadata.get("quality")
                            if quality in ["ok", "good", "excellent"]:
                                filtered_episode_paths.append(ep_path)
                                break
            except (OSError, ValueError, KeyError) as e:
                print(f"Error reading {ep_path}: {e}")
                continue

    return filtered_episode_paths

def filter_episode_paths(
    episode_paths: list[pathlib.Path],
    operator_blacklist: list[str],
    rating_whitelist: list[str],
) -> list[pathlib.Path]:
    """Filter episode paths based on operator blacklist and rating whitelist.

    Args:
        episode_paths: List of episode file paths
        operator_blacklist: List of operator names to ignore
        rating_whitelist: List of state names to include

    Returns:
        List of filtered episode paths
    """
    OP_INDEX = 1
    RATING_INDEX = 2

    if len(operator_blacklist) == 0 and len(rating_whitelist) == 0:
        return episode_paths

    filtered_paths = []
    for ep_path in episode_paths:
        parts = ep_path.stem.split("_")
        operator = parts[OP_INDEX]
        rating = parts[RATING_INDEX]

        if (not operator_blacklist or operator not in operator_blacklist) and (
            not rating_whitelist or rating in rating_whitelist
        ):
            filtered_paths.append(ep_path)

    return filtered_paths


def validate_input_dir(path: pathlib.Path) -> None:
    """Validate that input directory exists.

    Args:
        path: Directory path to validate

    Raises:
        FileNotFoundError: If directory doesn't exist
    """
    if not path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {path}")


def validate_output_dir(path: pathlib.Path, force: bool = False) -> None:
    """Validate output directory, handle --force flag.

    Args:
        path: Output directory path
        force: If True, delete existing directory

    Raises:
        FileExistsError: If directory exists and force=False
    """
    if path.exists():
        if force:
            print(
                f"Warning: Output directory {path} already exists. Overwriting due to --force."
            )
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {path}. Use --force to overwrite."
            )


def resolve_blacklist_path(blacklist_arg: pathlib.Path | None) -> pathlib.Path | None:
    """Resolve blacklist path with fallback logic.

    Handles both file paths and directory paths. If directory is provided,
    looks for blacklist.json in the directory or meta subdirectory.

    Args:
        blacklist_arg: Blacklist file or directory path

    Returns:
        Resolved blacklist file path, or None if not found
    """
    if blacklist_arg is None:
        return None

    if not blacklist_arg.exists():
        print(f"Warning: Specified blacklist file does not exist: {blacklist_arg}")
        return None

    if blacklist_arg.is_file():
        return blacklist_arg

    # Directory case - try to find blacklist.json
    blacklist_path = blacklist_arg / BLACKLIST_FILE
    if blacklist_path.exists():
        return blacklist_path

    blacklist_path = blacklist_arg / META_DIR / BLACKLIST_FILE
    if blacklist_path.exists():
        return blacklist_path

    print(f"Warning: No blacklist file found in directory: {blacklist_arg}")
    return None


def build_output_path(
    episodes_dir: pathlib.Path,
    output_parent: pathlib.Path | None,
    name_prefix: str | None,
    mode_suffix: str,
) -> pathlib.Path:
    """Construct output directory path.

    Args:
        episodes_dir: Input episodes directory
        output_parent: Parent directory for output (if None, uses episodes_dir/../../lerobot)
        name_prefix: Prefix for output directory name (if None, uses episodes_dir.name)
        mode_suffix: Mode string to append to directory name

    Returns:
        Constructed output directory path
    """
    if output_parent is None:
        output_parent = episodes_dir.parent.parent / "lerobot"

    if name_prefix is None:
        name_prefix = episodes_dir.name

    return output_parent / f"{name_prefix}_{mode_suffix}"


def copy_blacklist(
    output_dir: pathlib.Path, blacklist_path: pathlib.Path | None
) -> None:
    """Copy blacklist file to output directory.

    Args:
        output_dir: Output dataset directory
        blacklist_path: Source blacklist file path (can be None)
    """
    if blacklist_path is None:
        return

    if not blacklist_path.exists():
        return

    if blacklist_path.is_file():
        blacklist_dst = output_dir / META_DIR / BLACKLIST_FILE
        shutil.copy(blacklist_path, blacklist_dst)
        print(f"Copied blacklist to {blacklist_dst}")


def save_metadata(
    output_dir: pathlib.Path,
    episode_mapping: dict[int, str],
    blacklist: list[int],
    config,
) -> None:
    """Save all metadata files.

    Args:
        output_dir: Output dataset directory
        episode_mapping: Mapping of episode indices to file paths
        blacklist: List of blacklisted episode indices
        config: Pipeline configuration with to_dict() method
    """
    meta_dir = output_dir / META_DIR

    with open(meta_dir / EPISODE_MAPPING_FILE, "w", encoding="utf-8") as f:
        json.dump(episode_mapping, f, indent=2)

    with open(meta_dir / PIPELINE_CONFIG_FILE, "w", encoding="utf-8") as f:
        json.dump(config.to_dict(), f, indent=2)

    with open(meta_dir / BLACKLIST_FILE, "w", encoding="utf-8") as f:
        json.dump(blacklist, f, indent=2)

    print(f"\nMetadata saved to {meta_dir}")

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

"""Upload a trained policy checkpoint to the Hugging Face Hub.

Usage (CLI):
    python -m example_policies.data_ops.upload_model \
        --checkpoint /data/models/my_model \
        --repo-id pokeandwiggle/my_model \
        --private

Usage (Python):
    from example_policies.data_ops.upload_model import upload_model
    upload_model("/data/models/my_model", "pokeandwiggle/my_model", private=True)
"""

from __future__ import annotations

import pathlib

from huggingface_hub import HfApi


def upload_model(
    checkpoint_path: str | pathlib.Path,
    repo_id: str,
    *,
    private: bool = True,
) -> str:
    """Upload a policy checkpoint directory to the Hugging Face Hub.

    Args:
        checkpoint_path: Path to the training output or ``pretrained_model`` dir.
            If the path contains ``checkpoints/last/pretrained_model``, it will
            be resolved automatically.
        repo_id: HF repo id, e.g. ``"pokeandwiggle/my_model"``.
        private: Whether the repo should be private.

    Returns:
        The URL of the uploaded repo.
    """
    checkpoint_path = _resolve_checkpoint(pathlib.Path(checkpoint_path))

    api = HfApi()

    # Create repo (no-op if it already exists)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    print(f"Repo: https://huggingface.co/{repo_id}")

    # Upload the whole pretrained_model directory
    print(f"Uploading {checkpoint_path} ...")
    api.upload_folder(
        repo_id=repo_id,
        folder_path=str(checkpoint_path),
        repo_type="model",
    )

    url = f"https://huggingface.co/{repo_id}"
    print(f"✅ Upload complete: {url}")
    return url


def _resolve_checkpoint(path: pathlib.Path) -> pathlib.Path:
    """Resolve to the ``pretrained_model`` dir if needed."""
    if (path / "config.json").exists():
        return path
    candidate = path / "checkpoints" / "last" / "pretrained_model"
    if candidate.exists():
        return candidate
    raise FileNotFoundError(
        f"Cannot find config.json in {path} or {candidate}. "
        "Please point to the pretrained_model directory."
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Upload a policy checkpoint to Hugging Face.")
    parser.add_argument(
        "--checkpoint",
        type=pathlib.Path,
        required=True,
        help="Path to the training output or pretrained_model directory.",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="HF repo ID, e.g. 'pokeandwiggle/my_model'.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        default=True,
        help="Make the repo private (default: True).",
    )
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make the repo public.",
    )

    args = parser.parse_args()
    private = not args.public

    upload_model(args.checkpoint, args.repo_id, private=private)

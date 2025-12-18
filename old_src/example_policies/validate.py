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
from pathlib import Path

import torch
from lerobot.datasets.lerobot_dataset import LeRobotDataset

from example_policies.robot_deploy.policy_loader import load_policy


def main():
    parser = argparse.ArgumentParser(description="Robot service client")
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to the policy checkpoint directory.",
    )

    # Add argument dataset
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to the dataset directory.",
    )

    checkpoint_dir = parser.parse_args().checkpoint
    dataset_root_dir = parser.parse_args().dataset

    # Select your device
    device = "cpu" if not torch.cuda.is_available() else "cuda"

    policy, cfg = load_policy(checkpoint_dir)
    policy.to(device)

    dataset = LeRobotDataset(
        repo_id=dataset_root_dir,
        root=dataset_root_dir,
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=8,
        batch_size=1,
        shuffle=False,
        pin_memory=device != "cpu",
        drop_last=True,
    )

    for batch in dataloader:
        # Process each batch

        # roll random index
        rand_idx = torch.randint(0, len(dataloader), (1,)).item()
        second_batch = dataloader.dataset[rand_idx]

        for k in batch:
            if k.startswith("observation.images"):
                batch[k] = second_batch[k].unsqueeze(0)

        action = policy.select_action(batch)

        tgt = batch["action"].detach().float().view(-1)
        pred = action.detach().float().view(-1)
        diff = pred - tgt

        def _fmt(v: torch.Tensor, w=9, p=3):
            return f"{float(v):{w}.{p}f}"

        # Tabular print
        print("\nIdx |    Target |     Pred  |     Diff ")
        print("----+-----------+-----------+----------")
        for i, (t, p_, d) in enumerate(zip(tgt, pred, diff)):
            print(f"{i:3d} | {_fmt(t)} | {_fmt(p_)} | {_fmt(d)}")

        print(
            f"\nSummary: mae={diff.abs().mean():.4f} "
            f"max_abs={diff.abs().max():.4f} "
            f"rmse={(diff.pow(2).mean().sqrt()):.4f}"
        )

        input("Press Enter to continue...")


if __name__ == "__main__":
    main()

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

import logging
import os
import pathlib
import shutil

from lerobot.constants import PRETRAINED_MODEL_DIR
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies import factory as lerobot_factory
from lerobot.utils import train_utils, wandb_utils

from .policies.factory import get_policy
from .utils.constants import INFO_FILE, META_DIR


def monkey_patch_dataset():
    original_fn = LeRobotDataset._get_query_indices

    def patched_get_query_indices(
        self, idx: int, ep_idx: int
    ) -> tuple[dict[str, list[int | bool]]]:
        if self.episodes is not None:
            ep_idx = self.episodes.index(ep_idx)
        return original_fn(self, idx, ep_idx)

    LeRobotDataset._get_query_indices = patched_get_query_indices


def monkey_patch_policy_factory():
    original_lerobot_factory_fn = lerobot_factory.get_policy_class

    def extended_get_policy_cls(name: str):
        try:
            return original_lerobot_factory_fn(name)
        except NotImplementedError:
            return get_policy(name)
        except Exception as e:
            raise e

    lerobot_factory.get_policy_class = extended_get_policy_cls


def monkey_patch_save_checkpoint():
    original_save_fn = train_utils.save_checkpoint

    def patched_save_checkpoint(
        checkpoint_dir: pathlib.Path, step: int, cfg, policy, optimizer, scheduler
    ):
        original_save_fn(checkpoint_dir, step, cfg, policy, optimizer, scheduler)
        pretrained_dir = checkpoint_dir / PRETRAINED_MODEL_DIR
        meta_json = pathlib.Path(cfg.dataset.root) / META_DIR / INFO_FILE
        # Copy Metadata for deployment data selection
        shutil.copy(meta_json, pretrained_dir / "dataset_info.json")

    train_utils.save_checkpoint = patched_save_checkpoint


def monkey_patch_wandb():
    "Hacky way to resume training with allow instead of must"
    orig_class = wandb_utils.WandBLogger

    class PatchedWandBLogger(orig_class):
        """A helper class to log object using wandb."""

        def __init__(self, cfg):
            self.cfg = cfg.wandb
            self.log_dir = cfg.output_dir
            self.job_name = cfg.job_name
            self.env_fps = cfg.env.fps if cfg.env else None
            self._group = wandb_utils.cfg_to_group(cfg)

            # Set up WandB.
            os.environ["WANDB_SILENT"] = "True"
            import wandb

            wandb_run_id = (
                cfg.wandb.run_id
                if cfg.wandb.run_id
                else (
                    wandb_utils.get_wandb_run_id_from_filesystem(self.log_dir)
                    if cfg.resume
                    else None
                )
            )
            wandb.init(
                id=wandb_run_id,
                project=self.cfg.project,
                entity=self.cfg.entity,
                name=self.job_name,
                notes=self.cfg.notes,
                tags=wandb_utils.cfg_to_group(cfg, return_list=True),
                dir=self.log_dir,
                config=cfg.to_dict(),
                # TODO(rcadene): try set to True
                save_code=False,
                # TODO(rcadene): split train and eval, and run async eval with job_type="eval"
                job_type="train_eval",
                resume="allow" if cfg.resume else None,
                mode=(
                    self.cfg.mode
                    if self.cfg.mode in ["online", "offline", "disabled"]
                    else "online"
                ),
            )
            run_id = wandb.run.id
            # NOTE: We will override the cfg.wandb.run_id with the wandb run id.
            # This is because we want to be able to resume the run from the wandb run id.
            cfg.wandb.run_id = run_id
            # Handle custom step key for rl asynchronous training.
            self._wandb_custom_step_key: set[str] | None = None
            print(
                wandb_utils.colored(
                    "Logs will be synced with wandb.", "blue", attrs=["bold"]
                )
            )
            logging.info(
                f"Track this run --> {wandb_utils.colored(wandb.run.get_url(), 'yellow', attrs=['bold'])}"
            )
            self._wandb = wandb

    wandb_utils.WandBLogger = PatchedWandBLogger


def apply_patches():
    monkey_patch_policy_factory()
    monkey_patch_dataset()
    monkey_patch_save_checkpoint()
    monkey_patch_wandb()

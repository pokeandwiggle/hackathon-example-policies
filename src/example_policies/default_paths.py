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

"""Central default paths for all example_policies tooling.

All hard-coded filesystem defaults live here so they are easy to find,
override, and keep consistent across conversion, training, and deployment.

On farm stations ``/data`` is a mounted volume.  For other environments
override the paths via CLI flags or environment variables.
"""

import pathlib

# ── Root volume ──────────────────────────────────────────────────────
DATA_ROOT = pathlib.Path("/data")

# ── Dataset conversion ───────────────────────────────────────────────
# Default root for LeRobot datasets produced by dataset_conversion_synced.
# Individual datasets land at  DATASETS_DIR / "<task>_<operator>".
DATASETS_DIR = DATA_ROOT / "lerobot"

# ── Training ─────────────────────────────────────────────────────────
# Default root for training checkpoints / experiment directories.
MODELS_DIR = DATA_ROOT / "models"

# ── Deployment / rollout recordings ──────────────────────────────────
# Default root for deploy-loop rollout recordings.
ROLLOUT_RECORDINGS_DIR = DATA_ROOT / "rollout_recordings"

# ── Plots / reports ──────────────────────────────────────────────────
# Default output directory for quality-report PDFs and diagnostic plots.
PLOTS_DIR = DATA_ROOT / "plots"

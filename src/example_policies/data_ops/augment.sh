#!/bin/bash
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


# Check if correct number of arguments provided
if [ $# -ne 2 ]; then
    echo "Usage: $0 <input_path> <output_path>"
    echo "Example: $0 data/depth_raw data/aug_depth"
    exit 1
fi


# Get the directory of this script
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="$1"
OUTPUT_PATH="$2"

TMP_DIR=$(mktemp -d /tmp/lerobot_tmp_XXXXXX)
# Ensure temp directory is cleaned up on exit
cleanup() {
    rm -rf "$TMP_DIR"
}
trap cleanup EXIT
DATASET_CONVERSION="$SCRIPT_DIR/dataset_conversion.py"
MERGE_LEROBOT="$SCRIPT_DIR/merge_lerobot.py"

# Run dataset_conversion.py for each index
for idx in {0..5}; do
    echo "Running dataset_conversion.py with subsample_offset=$idx"
    python "$DATASET_CONVERSION" "$INPUT_PATH" --subsample-offset "$idx" --output "$TMP_DIR/tmp_$idx"
    echo "Finished dataset_conversion.py with subsample_offset=$idx"
    echo "----------------------------------------"
done

# Run merger script with all directories
echo "Running merge_lerobot.py"
python "$MERGE_LEROBOT" "$TMP_DIR/tmp_{0..5}" --output "$OUTPUT_PATH"
echo "Finished merge_lerobot.py"

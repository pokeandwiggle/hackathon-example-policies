# Create a Custom Dataset

This guide explains how to convert raw robot data from `.mcap` files into the LeRobot dataset format using the provided [conversion script](../notebooks/01_create_dataset.ipynb).

### Basic Usage

To convert a directory of `.mcap` episode files, run the following command. The script will recursively find all episodes, sort them by creation date, and process them into a new dataset directory.

```bash
python -m example_policies.data_ops.dataset_conversion <path-to-your-episodes> --output <path-to-output-dataset>
```

### Configuration

You can customize the dataset by adding command-line flags to the conversion command. For a full list of options, run the script with `--help` or refer to `example_policies.data_ops.pipeline_config`.

#### Observation Features

Control which data is included in the `observation` state.

-   `--include_joint_positions`: Include joint angles. Default is off.
-   `--include_joint_velocities`: Include joint velocities. Default is off.
-   `--include_tcp_poses`: Include end-effector poses. Default is on.
-   `--include_rgb_images`: Include RGB camera streams. Default is on.
-   `--include_depth_images`: Include depth camera streams. Default is off.

**Example:**
```bash
python -m ... --include_tcp_poses --include_rgb_images
```

#### Action Representation

Define the format of the `action` vector using the `--action_level` flag.

-   `delta_tcp` (Default): Change in TCP pose (`[d_xyz, d_rotvec, gripper]`).
-   `tcp`: Absolute TCP pose (`[xyz, quat, gripper]`).
-   `joint`: Absolute joint angles.
-   `delta_joint`: Change in joint angles.

**Example:**
```bash
python -m ... --action_level joint
```

#### Subsampling and Filtering

Control the frame rate and duration of the episodes.

-   `--target_fps`: The desired frames per second of the final dataset (default: 10).
-   `--min_episode_seconds`: Minimum duration for an episode to be included (default: 15).
-   `--max_pause_seconds`: Max duration of a standstill to keep before cutting it as a pause (default: 0.2).

**Example:**
```bash
python -m ... --target_fps 15 --min_episode_seconds 5
```

These dynamics settings highly depend on the robot task.

### Output Directory Structure

The [conversion script](../notebooks/01_create_dataset.ipynb) generates a directory with the following structure, compatible with the LeRobot framework.

```
<output-dataset>/
├── data/
│   └── ... (Parquet files with chunked data)
│
├── meta/
│   ├── blacklist.json        # Episodes too short to be valid training data.
│   ├── episode_mapping.json  # Maps original recording names to dataset episode indices.
│   ├── pipeline_config.json  # Conversion settings for reproducibility.
│   └── ... (Other dataset metadata files)
│
└── videos/
    └── chunk-000/
        ├── observation.images.rgb_left/
        ├── observation.images.rgb_right/
        └── observation.images.rgb_static/

```

Please note that our dataset structure saves additional metadata originally not included in the LeRobot dataset format.
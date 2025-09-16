# Dataset Upload and Visualization Guide

> [!NOTE]
> **TL;DR**: Use `upload.py` to push datasets to Hugging Face Hub and `visualize.py` to view episodes locally or from the hub. Quick commands:
> - Upload: `python src/example_policies/data_ops/upload.py --repo-id username/dataset --root /path/to/dataset`
> - Visualize: `python src/example_policies/data_ops/visualize.py --repo-id username/dataset` or add `--local --root /path/to/dataset` for local datasets

This guide explains how to use the dataset upload and visualization tools provided in the `src/example_policies/data_ops/` directory.

## Overview

The project provides two main utilities for working with LeRobot datasets:
- **Upload Tool** (`upload.py`): Upload local datasets to Hugging Face Hub
- **Visualization Tool** (`visualize.py`): Visualize datasets either from local storage or Hugging Face Hub

## Upload Tool

### Purpose
Upload a local LeRobot dataset to Hugging Face Hub for sharing and remote access.

### Usage
```bash
python src/example_policies/data_ops/upload.py --repo-id <repo-id> --root <local-path>
```

### Parameters
- `--repo-id` (required): Hugging Face repository ID where the dataset will be uploaded
  - Format: `username/dataset_name`
  - Example: `myuser/robot_demonstrations`
- `--root` (required): Local path to the dataset directory

### Example
```bash
python src/example_policies/data_ops/upload.py \
    --repo-id "myuser/pickup_demonstrations" \
    --root "/path/to/local/dataset"
```

### Prerequisites
- Ensure you're authenticated with Hugging Face Hub (`huggingface-cli login`)
- Have write permissions to the target repository
- Dataset must be in LeRobot format

## Visualization Tool

### Purpose
Visualize episodes from LeRobot datasets, supporting both local and remote (Hugging Face Hub) datasets.

### Usage

#### Visualize from Hugging Face Hub
```bash
python src/example_policies/data_ops/visualize.py --repo-id <repo-id> [--episode-index <index>]
```

#### Visualize from Local Directory
```bash
python src/example_policies/data_ops/visualize.py --local --root <local-path> [--episode-index <index>]
```

### Parameters
- `--repo-id`: Hugging Face repository ID to load dataset from
  - Format: `username/dataset_name`
  - Required when not using `--local`
- `--local`: Flag to indicate loading from local directory
- `--root`: Local path to dataset directory
  - Required when using `--local`
- `--episode-index`: Episode number to visualize (default: 0)

### Examples

#### Visualize from Hub
```bash
# Visualize episode 0 from a public dataset
python src/example_policies/data_ops/visualize.py \
    --repo-id "lerobot/pusht"

# Visualize episode 5 from a specific dataset
python src/example_policies/data_ops/visualize.py \
    --repo-id "myuser/custom_dataset" \
    --episode-index 5
```

#### Visualize Local Dataset
```bash
# Visualize local dataset episode 0
python src/example_policies/data_ops/visualize.py \
    --local \
    --root "/path/to/local/dataset"

# Visualize specific episode from local dataset
python src/example_policies/data_ops/visualize.py \
    --local \
    --root "/path/to/local/dataset" \
    --episode-index 3
```

## Common Workflow

### 1. Visualize Locally (Optional)
Before uploading, verify your dataset by visualizing it locally:
```bash
python src/example_policies/data_ops/visualize.py \
    --local \
    --root "/path/to/your/dataset"
```

### 2. Upload to Hub
Upload the dataset to Hugging Face Hub:
```bash
python src/example_policies/data_ops/upload.py \
    --repo-id "your-username/your-dataset-name" \
    --root "/path/to/your/dataset"
```

### 3. Verify Upload
Visualize the uploaded dataset to ensure it was uploaded correctly:
```bash
python src/example_policies/data_ops/visualize.py \
    --repo-id "your-username/your-dataset-name"
```

This will visualize the dataset in rerun.
You can visualize the dataset online with online tool: https://huggingface.co/spaces/lerobot/visualize_dataset


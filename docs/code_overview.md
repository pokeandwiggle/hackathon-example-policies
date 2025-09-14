# Code Structure Overview

This document provides a high-level overview of the `example_policies` package. The source code is organized into several key directories, each with a specific responsibility.

## Key Executable Scripts

For convenience, here are the primary scripts you will interact with:

*   `example_policies/train.py`: Starts a new training job.
*   `example_policies/data_ops/dataset_conversion.py`: Converts raw data into the `lerobot` format.
*   `example_policies/data_ops/merge_lerobot.py`: Merges multiple datasets.
*   `example_policies/robot_deploy/deploy.py`: Deploys a trained policy to the robot.

---

## Package Details

### `example_policies/`

This is the core package containing the primary entry points and configuration for training and evaluation.

*   **`train.py`**: The main script to start a training job.
*   **`validate.py`**: A script for running policy validation against offline training data.
*   **`config_factory.py`**: Manages and creates default policy configurations.
*   **`lerobot_patches.py`**: Applies necessary modifications to the underlying `lerobot` library.

### `data_ops/`

This package contains all tools and scripts related to dataset creation and manipulation.

*   **`dataset_conversion.py`**: Handles the conversion of raw data (e.g., rosbags) into the standardized `lerobot` dataset format.
*   **`merge_lerobot.py`**: Provides functionality to merge multiple `lerobot` datasets into one.
*   **`visualize.py`**: Includes tools for visualizing and inspecting `lerobot` datasets to verify correctness.
*   **`config/pipeline_config.py`**: Defines the configuration for the dataset conversion pipeline.

### `policies/`

This directory defines the neural network architectures and loss functions for the robot policies.

*   **`factory.py`**: A factory class to instantiate different policy classes.
*   **`models/`**: Contains the PyTorch implementations of the policy networks.
*   **`losses/`**: Defines custom loss functions, most importantly SO(3)-aware losses for working with TCP coordinates.

### `robot_deploy/`

This package contains the necessary code to execute a trained policy on the physical robot hardware.

*   **`deploy.py`**: The main entry point for running a policy on the robot.
*   **`policy_loader.py`**: Handles loading the trained model weights for inference.
*   **`action_translator.py`**: Translates the neural network's output into low-level robot commands.
*   **`robot_io/`**: Contains modules for interfacing with the robot's hardware.

### `training/`

This package provides custom training loops and utilities that extend the base `lerobot` training framework. It serves as a starting point for advanced users who wish to implement more complex training logic.

*   **`train_custom.py`**: Implements a customized training loop with project-specific logic.
*   **`utils.py`**: A collection of helper functions and utilities to support the training process.
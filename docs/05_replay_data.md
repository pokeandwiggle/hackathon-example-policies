# Data Replay Guide

> [!NOTE]
> This guide shows how to replay recorded data on the robot using the replay functionality.
>
> ```bash
> python -m example_policies.robot_deploy.debug_helpers.replay_data ./my_dataset \
>    --episode 2 \
>    --replay-frequency 10.0 \
>     --continuous-replay
> ```


## Overview

The data replay script allows you to execute previously recorded actions on the robot, which is useful for:
- Testing robot behavior with known good data
- Debugging action execution pipelines
- Validating data quality before training
- Demonstrating recorded behaviors

## Basic Usage

### Command Line Interface

```bash
python -m example_policies.robot_deploy.debug_helpers.replay_data <data_dir> [options]
```

### Required Arguments

- `data_dir`: Path to the LeRobot dataset directory containing episodes

### Optional Arguments

- `--server`: Robot service server address (default: localhost:50051)
- `--episode`: Episode index to replay (default: 0)
- `--replay-frequency`: Playback frequency in Hz (default: 5.0)
- `--continuous-replay`: Run continuously without user prompts (default: False)

## Examples

### Interactive Replay
Replay episode 0 with manual confirmation at each step:
```bash
python -m example_policies.robot_deploy.debug_helpers.replay_data ./my_dataset --episode 0
```

### Continuous Replay
Replay episode 2 at 10Hz without prompts:
```bash
python -m example_policies.robot_deploy.debug_helpers.replay_data ./my_dataset \
    --episode 2 \
    --replay-frequency 10.0 \
    --continuous-replay
```

### Remote Server
Replay on a remote robot server:
```bash
python -m example_policies.robot_deploy.debug_helpers.replay_data ./my_dataset \
    --server 192.168.1.100:50051
```

## Important Notes

- **Delta Actions Only**: Currently only supports delta TCP action mode
- **Manual Start**: You'll be prompted to position the robot at the start pose
- **Frequency**: Lower frequencies (1-5 Hz) are recommended to start with, increase later when you feel confident.

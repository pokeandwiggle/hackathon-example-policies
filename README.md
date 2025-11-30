<!-- <p align="center">
  <img alt="Repo Banner" src="docs/assets/banner.png" width="100%">
</p> -->

# Poke&Wiggle Policy Starter Kit

<p align="center">
  <a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
  <a href="https://www.python.org/downloads/"><img alt="Python versions" src="https://img.shields.io/pypi/pyversions/lerobot"></a>
  <a href="https://github.com/huggingface/lerobot"><img alt="Powered by LeRobot" src="https://img.shields.io/badge/Powered%20by-LeRobot-orange"></a>
</p>

This starter kit provides all the essential tools to get you from raw recoding data to a deployed policy on a real robot.

## 🚀 Installation

First, make sure you are logged into jupyterhub and are in a terminal window.

Then, clone this repo:

```
git clone https://github.com/pokeandwiggle/hackathon-example-policies.git && cd hackathon-example-policies
```

To install the required packages, simply run:

```sh
pip install -e .
```

Create a symblic link to the dataset directory by running:
```
ln -s /data data
```

If you have a Weights&Biases account for logging your policy training, log in using:
```
wandb login
```

## 🏁 Quickstart: Your First Policy

After recoding demonstrations, follow these steps to train and deploy a policy.

### 1. Convert Dataset

First, convert your recorded raw recoding data (ROS2 MCAP files) into a training dataset (LeRobot Dataset Format). This format is optimized for training. Use the jupyter notebook [01_create_dataset.ipynb](notebooks/01_create_dataset.ipynb).

<!-- ```bash
python src/example_policies/data_ops/data_conversion.py --episodes-dir <path/to/mcap_files_dir> --output <path/to/output_dataset_dir>
``` -->

### 2. Train a Policy

Next, train your policy. 
Use the jupyter notebook [02_train_model_simple.ipynb](notebooks/02_train_model_simple.ipynb).

<!-- 1.  **Configure:** Edit `src/example_policies/train.py` and `src/example_policies/config_factory.py` to select your model architecture, dataset path, and hyperparameters.
2.  **Run Training:** Execute the training script from the project root.

```bash
python src/example_policies/train.py [DATASET ROOT DIR] [--batch_size 32]
``` -->

### 3. Deploy on the Robot

Finally, see your robot policy come to life! Deploy your trained policy checkpoint to the real robot. Use the jupyter notebook [03_deploy_policy.ipynb](notebooks/03_deploy_policy.ipynb).

<!-- -   `--checkpoint`: Path to the trained model checkpoint directory.
-   `--server`: IP address and port of the robot's gRPC service.

```bash
python src/example_policies/robot_deploy/deploy.py --checkpoint <path/to/checkpoint> --server <ip:port>
``` -->


## 🗺️ What's in this repo?

Here's a quick overview of the key directories:

-   `src/example_policies/data_ops/`: Scripts for converting your ROS2 bag files into the LeRobot format.
-   `src/example_policies/policies/`: Example policy implementations to build upon.
-   `src/example_policies/train.py`: The main script for training your policy.
-   `src/example_policies/robot_deploy/`: Scripts to deploy your trained policy onto the robot.
-   `docs/`: In-depth documentation for advanced usage.


## 🛠️ Resources & Support

-   **Documentation:** For advanced topics, check out the [`docs`](./docs) directory.
-   **LeRobot:** This project is built on the powerful [LeRobot](https://github.com/huggingface/lerobot) library. Their documentation is an excellent resource.

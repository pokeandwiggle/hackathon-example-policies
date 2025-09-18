from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig
from typing import Any
from dataclasses import dataclass, field
from functools import cached_property

from example_policies.robot_deploy.robot_io.robot_service import robot_service_pb2_grpc
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.policy_loader import load_policy
from example_policies.robot_deploy.action_translator import ActionTranslator
from lerobot.cameras import CameraConfig
import torch
import numpy as np
import logging
from example_policies.robot_deploy.debug_helpers.utils import print_info

logger = logging.getLogger(__name__)


@CameraConfig.register_subclass("robot_io_camera")
@dataclass
class RobotIOCameraConfig(CameraConfig):
    pass


import grpc


@RobotConfig.register_subclass("robot_io")
@dataclass
class RobotIOConfig(RobotConfig):
    host: str = "127.0.0.1"
    port: int = 50051

    checkpoint: str = ""

    # cameras
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Path to URDF file for kinematics
    # NOTE: It is highly recommended to use the urdf in the SO-ARM100 repo:
    # https://github.com/TheRobotStudio/SO-ARM100/blob/main/Simulation/SO101/so101_new_calib.urdf
    # urdf_path: str | None = None

    # End-effector frame name in the URDF
    # target_frame_name: str = "gripper_frame_link"

    # Default bounds for the end-effector position (in meters)
    end_effector_bounds: dict[str, list[float]] = field(
        default_factory=lambda: {
            "min": [-1.0, -1.0, -1.0],  # min x, y, z
            "max": [1.0, 1.0, 1.0],  # max x, y, z
        }
    )

    max_gripper_pos: float = 50

    end_effector_step_sizes: dict[str, float] = field(
        default_factory=lambda: {
            "x": 0.001,
            "y": 0.001,
            "z": 0.001,
        }
    )


class RobotIO(Robot):
    config_class = RobotIOConfig
    name = "robot_io"

    def __init__(self, config: RobotIOConfig):
        super().__init__(config)
        self.config = config

        server = f"{self.config.host}:{self.config.port}"
        channel = grpc.insecure_channel(server)
        service_stub = robot_service_pb2_grpc.RobotServiceStub(channel)
        checkpoint = self.config.checkpoint
        policy, cfg = load_policy(checkpoint)
        self.cfg = cfg
        # policy.to(device)
        robot_interface = RobotInterface(service_stub, cfg)
        model_to_action_trans = ActionTranslator(cfg)

        self.robot_interface = robot_interface
        self.model_to_action_trans = model_to_action_trans
        print(policy.config.input_features)
        # observation = robot_interface.get_observation(cfg.device, show=False)
        self.cameras = {"rgb_static": {}, "rgb_left": {}, "rgb_right": {}}
        self.state_feature_names = (
            robot_interface.observation_builder.state_feature_names
        )

    @property
    def _motors_ft(self) -> dict[str, type]:
        return {
            f"{state_feature_name}": float
            for state_feature_name in self.state_feature_names
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
        # ['observation.images.rgb_static', 'observation.images.rgb_left', 'observation.images.rgb_right']
        return {
            cam: (self.config.cameras[cam].height, self.config.cameras[cam].width, 3)
            for cam in self.cameras
        }

    @cached_property
    def observation_features(self) -> dict[str, type | tuple]:
        return {**self._motors_ft, **self._cameras_ft}

    @property
    def action_features(self) -> dict[str, Any]:
        """
        Define action features for end-effector control.
        Returns dictionary with dtype, shape, and names.
        """
        return {
            "dtype": "float32",
            "shape": (4,),
            "names": {"delta_x": 0, "delta_y": 1, "delta_z": 2, "gripper": 3},
        }

    def calibrate(self) -> None:
        pass

    def connect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:

        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")
        observation = self.robot_interface.get_observation(self.cfg.device, show=False)
        self.current_observation = observation
        obs_dict = {}

        if observation:
            for i, name in enumerate(self.state_feature_names):
                obs_dict[name] = observation["observation.state"][0, i].item()

            # Process camera images
            for cam_key in self.cameras.keys():
                rgb_key = f"observation.images.{cam_key}"
                if rgb_key in self.cfg.input_features.keys():
                    # cfg_shape = self.cfg.input_features[img_key].shape
                    img_data = observation[f"observation.images.{cam_key}"].to("cpu")
                    img_data = img_data.permute(0, 2, 3, 1)  # BCHW to BHWC
                    img_array = img_data.squeeze(0)  # .numpy()
                    # img_array = (img_array * 255).astype(np.uint8)
                    img_array = (img_array * 255).to(dtype=torch.uint8)
                    img_array = img_array.numpy()
                    obs_dict[cam_key] = img_array

        return obs_dict

    def get_action(self) -> dict[str, Any]:
        return {}

    def send_action(self, action: dict[str, Any]) -> None:
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")
        # Convert action to numpy array if not already

        if isinstance(action, dict):
            if all(k in action for k in ["delta_x", "delta_y", "delta_z"]):
                delta_ee = np.array(
                    [
                        action["delta_x"] * self.config.end_effector_step_sizes["x"],
                        action["delta_y"] * self.config.end_effector_step_sizes["y"],
                        action["delta_z"] * self.config.end_effector_step_sizes["z"],
                    ],
                    dtype=np.float32,
                )
                if "gripper" not in action:
                    action["gripper"] = [1.0]
                action = np.append(delta_ee, action["gripper"])
            else:
                logger.warning(
                    f"Expected action keys 'delta_x', 'delta_y', 'delta_z', got {list(action.keys())}"
                )
                action = np.zeros(4, dtype=np.float32)

        # # Add delta to position and clip to bounds
        # desired_ee_pos[:3, 3] = self.current_ee_pos[:3, 3] + action[:3]
        # if self.end_effector_bounds is not None:
        #     desired_ee_pos[:3, 3] = np.clip(
        #         desired_ee_pos[:3, 3],
        #         self.end_effector_bounds["min"],
        #         self.end_effector_bounds["max"],
        #     )
        action = torch.tensor(
            [
                [
                    action[0],
                    action[1],
                    action[2],
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,
                    0,          
                ]
            ],
            device="cuda:0",
            dtype=torch.float32,
        )
        print(f"\n=== RAW MODEL PREDICTION ===")
        print_info(0, self.current_observation, action)

        action = self.model_to_action_trans.translate(action, self.current_observation)
        print(f"\n=== ABSOLUTE ROBOT COMMANDS ===")
        print_info(0, self.current_observation, action)

        # self.robot_interface.send_action(action, self.model_to_action_trans.action_mode)

    # configure
    def configure(self) -> None:
        pass

    def disconnect(self) -> None:
        pass

    @property
    def is_calibrated(self) -> bool:
        return True

    @property
    def is_connected(self) -> bool:
        return True

    def reset(self) -> None:
        pass

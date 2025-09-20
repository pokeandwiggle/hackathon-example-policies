from lerobot.robots.robot import Robot
from lerobot.robots.config import RobotConfig
from typing import Any
from dataclasses import dataclass, field
from functools import cached_property

from example_policies.robot_deploy.robot_io.robot_service import robot_service_pb2_grpc
from example_policies.robot_deploy.robot_io.robot_interface import RobotInterface
from example_policies.robot_deploy.policy_loader import load_policy_config
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
            "l_x": 0.001,
            "l_y": 0.001,
            "l_z": 0.001,
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

        # TODO: cfg should not be loeaded from checkpoint, maybe from config
        checkpoint = self.config.checkpoint
        cfg = load_policy_config(checkpoint)
        self.cfg = cfg
        # print(self.cfg.input_features)
        robot_interface = RobotInterface(service_stub, cfg)
        model_to_action_trans = ActionTranslator(cfg)

        self.robot_interface = robot_interface
        self.model_to_action_trans = model_to_action_trans

        # observation = robot_interface.get_observation(cfg.device, show=False)
        # self.cameras = {"rgb_static": {}, "rgb_left": {}, "rgb_right": {}}
        self.cameras = {cam: {} for cam in config.cameras.keys()}
        self.state_feature_names = (
            robot_interface.observation_builder.state_feature_names
        )

        # Store the bounds for end-effector position
        self.end_effector_bounds = self.config.end_effector_bounds

        self.current_joint_pos = None
        # self.current_observation = None

    # @property
    # def _motors_ft(self) -> dict[str, type]:
    #     return {
    #         f"{state_feature_name}": float
    #         for state_feature_name in self.state_feature_names
    #     }
    @property
    def _motors_ft(self) -> dict:
        state_names = self.state_feature_names
        return {
            # "action": {
            #     "dtype": "float32",
            #     "shape": (len(action_names),),
            #     "names": action_names,
            # },
            "observation.state": {
                "dtype": "float32",
                "shape": (len(state_names),),
                "names": state_names,
            },
        }

    @property
    def _cameras_ft(self) -> dict[str, tuple]:
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
            "shape": (8,),
            "names": {"l_delta_x": 0, "l_delta_y": 1, "l_delta_z": 2, "r_delta_x": 3, "r_delta_y": 4, "r_delta_z": 5, "l_gripper": 6, "r_gripper": 7},
        }

    def calibrate(self) -> None:
        pass

    def connect(self) -> None:
        pass

    def get_observation(self) -> dict[str, Any]:
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")

        observation = self.robot_interface.get_observation(self.cfg.device, show=False)
        # observation.images.rgb_static [1, 3, 640, 640]
        # observation.images.rgb_left [1, 3, 640, 640]
        # observation.images.rgb_right [1, 3, 640, 640]
        # observation.observation.state [1, 32]
        self.current_observation = observation
        obs_dict = {}

        if observation:
            # for i, name in enumerate(self.state_feature_names):
            #     obs_dict[name] = observation["observation.state"][0, i].item()
            obs_dict["observation.state"] = observation["observation.state"].squeeze(0).cpu().numpy()

            # Process camera images
            for cam_key in self.cameras.keys():
                rgb_key = f"observation.images.{cam_key}"
                if rgb_key in self.cfg.input_features.keys():
                    # cfg_shape = self.cfg.input_features[img_key].shape
                    img_data = observation[f"observation.images.{cam_key}"]
                    img_data = img_data.permute(0, 2, 3, 1)  # BCHW to BHWC
                    img_array = img_data.squeeze(0) # Remove batch dim
                    img_array = (img_array * 255).to(dtype=torch.uint8)
                    img_array = img_array.cpu().numpy()
                    obs_dict[cam_key] = img_array

        return obs_dict

    def get_action(self) -> dict[str, Any]:
        return {}

    def send_action(self, action: dict[str, Any]) -> None:
        # if not self.is_connected:
        #     raise DeviceNotConnectedError(f"{self} is not connected.")
        # Convert action to numpy array if not already

        if isinstance(action, dict):
            # current_observation = self.current_observation
            current_observation = {"observation.state": torch.from_numpy(action["current_observation"]["observation.state"]).unsqueeze(0).to("cuda:0").float()}
            # current_observation["observation.state"] = current_observation["observation.state"].expand_dims(0)

            if self.current_joint_pos is None:
                # self.current_joint_pos = np.ones(2, dtype=np.float32)
                self.current_joint_pos = np.array(
                    [
                        np.clip(action["l_gripper"], 0, 2),
                        np.clip(action["r_gripper"], 0, 2),
                    ],
                    dtype=np.float32,
                )
            del action["current_observation"]

            if all(k in action for k in ["l_delta_x", "l_delta_y", "l_delta_z", "r_delta_x", "r_delta_y", "r_delta_z", "l_gripper", "r_gripper"]):
                delta_ee = np.array(
                    [
                        action["l_delta_x"] * self.config.end_effector_step_sizes["l_x"],
                        action["l_delta_y"] * self.config.end_effector_step_sizes["l_y"],
                        action["l_delta_z"] * self.config.end_effector_step_sizes["l_z"],
                        action["r_delta_x"] * self.config.end_effector_step_sizes["r_x"],
                        action["r_delta_y"] * self.config.end_effector_step_sizes["r_y"],
                        action["r_delta_z"] * self.config.end_effector_step_sizes["r_z"],
                    ],
                    dtype=np.float32,
                )

                # TODO: enable right arm control
                # delta_ee = np.append(
                #     delta_ee,
                #     np.array(
                #         [
                #             action["r_delta_x"]
                #             * self.config.end_effector_step_sizes["r_x"],
                #             action["r_delta_y"]
                #             * self.config.end_effector_step_sizes["r_y"],
                #             action["r_delta_z"]
                #             * self.config.end_effector_step_sizes["r_z"],
                #         ],
                #         dtype=np.float32,
                #     ),
                # )
                # delta_ee = np.append(delta_ee, np.zeros(3, dtype=np.float32))

                # if "l_gripper" not in action:
                #     action["l_gripper"] = [1.0]
                # if "r_gripper" not in action:
                #     action["r_gripper"] = [1.0]

                delta_ee = np.append(delta_ee, action["l_gripper"])
                delta_ee = np.append(delta_ee, action["r_gripper"])
                action = delta_ee
            else:
                raise ValueError(
                    f"Expected action keys 'l_delta_x', 'l_delta_y', 'l_delta_z', 'l_gripper', got {list(action.keys())}"
                )
                # action = np.zeros(8, dtype=np.float32)
        else:
            raise ValueError(f"Expected action to be a dict, got {type(action)}")

        # joint_action["gripper.pos"] = np.clip(
        #     self.current_joint_pos + (action[-2:] - 1) * self.config.max_gripper_pos,
        #     5,
        #     self.config.max_gripper_pos,
        # )
        # import pdb; pdb.set_trace()self.current_joint_pos = np.array([joint_action["gripper.pos"][0], joint_action["gripper.pos"][1]], dtype=np.float32)

        # print action

        action[6:] = np.clip(action[6:], 0, 2)
        print(action[6:])
        print(f"Current gripper pos:\n{self.current_joint_pos}")    
        self.current_joint_pos = np.array(
            [
                self.current_joint_pos[0] + (action[6] - 1),
                self.current_joint_pos[1] + (action[7] - 1),
            ],
            dtype=np.float32,
        )
        print(f"New gripper pos:\n{self.current_joint_pos}")
        self.current_joint_pos = np.clip(self.current_joint_pos, 0, 2)
        print(f"New gripper pos:\n{self.current_joint_pos}")

        # TODO: add delta rotation
        action = torch.tensor(
            [
                [
                    action[0],
                    action[1],
                    action[2],
                    0,
                    0,
                    0,
                    # action[3],
                    # action[4],
                    # action[5],
                    0,0,0,
                    0,
                    0,
                    0,
                    self.current_joint_pos[0]/2.0,
                    self.current_joint_pos[1]/2.0,
                ]
            ],
            device="cuda:0",
            dtype=torch.float32,
        )

        # clip action
        action[:, :12] = torch.clamp(action[:, :12], -0.01, 0.01)
        # action[:, 12:] = torch.clamp(action[:, 12:], -1, 1)

        print(f"\n=== RAW MODEL PREDICTION ===")
        print_info(0, current_observation, action)

        action = self.model_to_action_trans.translate(action, current_observation)
        print(f"\n=== ABSOLUTE ROBOT COMMANDS ===")
        print_info(0, self.current_observation, action)

        # Absolute
        action[0, 0:3] = torch.clamp(
            action[0, 0:3],
            torch.tensor(self.end_effector_bounds["l_min"], device=action.device),
            torch.tensor(self.end_effector_bounds["l_max"], device=action.device),
        )
        action[0, 7:10] = torch.clamp(
            action[0, 7:10],
            torch.tensor(self.end_effector_bounds["r_min"], device=action.device),
            torch.tensor(self.end_effector_bounds["r_max"], device=action.device),
        )
        print(f"\n=== ABSOLUTE ROBOT COMMANDS CLAMPED ===")
        print_info(0, self.current_observation, action)

        # l_pos_a = action[0, 0:3]
        # l_quat_a = action[0, 3:7]
        # r_pos_a = action[0, 7:10]
        # r_quat_a = action[0, 10:14]
        # grips = action[0, 14:16]

        self.robot_interface.send_action(action, self.model_to_action_trans.action_mode)

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
        self.current_joint_pos = None

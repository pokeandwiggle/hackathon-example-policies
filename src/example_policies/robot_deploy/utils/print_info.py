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

import torch

from example_policies.utils.action_order import ActionMode
from example_policies.utils.constants import OBSERVATION_STATE

ACTION_MODE_STATE_MAP = {
    ActionMode.TCP: [f"tcp_left_pos_{i}" for i in "xyz"]
    + [f"tcp_left_quat_{i}" for i in "xyzw"]
    + [f"tcp_right_pos_{i}" for i in "xyz"]
    + [f"tcp_right_quat_{i}" for i in "xyzw"],
    ActionMode.TELEOP: [f"tcp_left_pos_{i}" for i in "xyz"]
    + [f"tcp_left_quat_{i}" for i in "xyzw"]
    + [f"tcp_right_pos_{i}" for i in "xyz"]
    + [f"tcp_right_quat_{i}" for i in "xyzw"],
    ActionMode.DELTA_TCP: [f"tcp_left_pos_{i}" for i in "xyz"]
    + [f"tcp_left_quat_{i}" for i in "xyzw"]
    + [f"tcp_right_pos_{i}" for i in "xyz"]
    + [f"tcp_right_quat_{i}" for i in "xyzw"],
    ActionMode.JOINT: [f"joint_pos_left_{i}" for i in range(7)]
    + [f"joint_pos_right_{i}" for i in range(7)],
    ActionMode.DELTA_JOINT: [f"joint_pos_left_{i}" for i in range(7)]
    + [f"joint_pos_right_{i}" for i in range(7)],
}

ACTION_MODE_ACTION_MAP = {
    ActionMode.TCP: [f"tcp_left_pos_{i}" for i in "xyz"]
    + [f"tcp_left_quat_{i}" for i in "xyzw"]
    + [f"tcp_right_pos_{i}" for i in "xyz"]
    + [f"tcp_right_quat_{i}" for i in "xyzw"]
    + ["gripper_left", "gripper_right"],
    ActionMode.TELEOP: [f"tcp_left_pos_{i}" for i in "xyz"]
    + [f"tcp_left_quat_{i}" for i in "xyzw"]
    + [f"tcp_right_pos_{i}" for i in "xyz"]
    + [f"tcp_right_quat_{i}" for i in "xyzw"]
    + ["gripper_left", "gripper_right"],
    ActionMode.DELTA_TCP: [f"tcp_left_dpos_{i}" for i in "xyz"]
    + [f"tcp_left_daa_{i}" for i in "xyz"]
    + [f"tcp_right_dpos_{i}" for i in "xyz"]
    + [f"tcp_right_daa_{i}" for i in "xyz"]
    + ["gripper_left", "gripper_right"],
    ActionMode.JOINT: [f"joint_pos_left_{i}" for i in range(7)]
    + [f"joint_pos_right_{i}" for i in range(7)]
    + ["gripper_left", "gripper_right"],
    ActionMode.DELTA_JOINT: [f"joint_pos_left_{i}" for i in range(7)]
    + [f"joint_pos_right_{i}" for i in range(7)]
    + ["gripper_left", "gripper_right"],
}


def _format_tensor(tensor, precision: int = 3):
    if isinstance(tensor, torch.Tensor):
        tensor = tensor.detach().cpu()

    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in tensor.tolist()) + "]"


def build_obs_indices(cfg, feature_filters: list[str]):
    """
    Build a list of indices for observation features that match any of the given filter strings.
    Args:
        cfg: Configuration object containing metadata about observation features.
        feature_filters (list[str]): List of substrings to filter feature names.

    Returns:
        list[int]: Indices of features whose names contain any of the filter strings.
    """
    names: list[str] = cfg.metadata["features"][OBSERVATION_STATE]["names"]
    indices = [i for i, n in enumerate(names) if any(f in n for f in feature_filters)]
    return indices


class InfoPrinter:
    def __init__(self, cfg) -> None:
        self.action_mode = ActionMode.parse_action_mode(cfg)
        self.obs_indices = build_obs_indices(
            cfg, ACTION_MODE_STATE_MAP[self.action_mode]
        )

    def print(
        self,
        step: int,
        observation: dict,
        action: torch.Tensor,
        raw_action: bool = True,
    ):
        obs = observation.get(OBSERVATION_STATE)
        if obs is not None and isinstance(obs, torch.Tensor) and obs.ndim == 2:
            obs = obs[0, self.obs_indices]
        if raw_action:
            if self.action_mode == ActionMode.DELTA_TCP:
                print_tcp_delta(step, obs, action)
                return
            if self.action_mode == ActionMode.DELTA_JOINT:
                print_joint_delta(step, obs, action)
                return
        if self.action_mode in (
            ActionMode.TCP,
            ActionMode.TELEOP,
            ActionMode.DELTA_TCP,
        ):
            print_tcp_abs(step, obs, action)
            return
        if self.action_mode in (ActionMode.JOINT, ActionMode.DELTA_JOINT):
            print_joint_abs(step, obs, action)
            return
        print(f"=== Step {step} | Unknown Action Mode {self.action_mode} ===")


def print_joint_abs(step: int, state: torch.Tensor, action: torch.Tensor):
    """
    Pretty print action vs (optional) observation.
    Formats:
      - Absolute Joint (16): L pos(7) + R pos(7) + 2 grippers
      - Delta Joint (16):  L dpos(7) + R dpos(7) + L grip + R grip
    """

    print(f"=== Step {step} | Absolute Joints ===")
    # Absolute Joint
    l_pos_a = action[0, 0:7]
    r_pos_a = action[0, 7:14]
    grips = action[0, 14:16]

    l_pos_o = state[0:7]
    r_pos_o = state[7:14]
    print(
        " Left  Joints | Obs", _format_tensor(l_pos_o), " Pred", _format_tensor(l_pos_a)
    )
    print(
        " Right Joints | Obs", _format_tensor(r_pos_o), " Pred", _format_tensor(r_pos_a)
    )
    print(" Grippers (L,R)", _format_tensor(grips))


def print_joint_delta(step: int, state: torch.Tensor, action: torch.Tensor):
    """
    Pretty print action vs (optional) observation.
    Formats:
      - Absolute Joint (16): L pos(7) + R pos(7) + 2 grippers
      - Delta Joint (16):  L dpos(7) + R dpos(7) + L grip + R grip
    """

    print(f"=== Step {step} | Delta Joints ===")
    # Delta Joint
    l_dpos = action[0, 0:7]
    r_dpos = action[0, 7:14]
    l_grip = action[0, 14]
    r_grip = action[0, 15]

    print(" Left  ΔJoints", _format_tensor(l_dpos))
    print(" Right ΔJoints", _format_tensor(r_dpos))
    print(f" Grippers    [L={float(l_grip):.3f}, R={float(r_grip):.3f}]")


def print_tcp_abs(step: int, state: torch.Tensor, action: torch.Tensor):
    """
    Pretty print action vs (optional) observation.
    Formats:
      - Absolute TCP (16): L pos(3) + L quat(4) + R pos(3) + R quat(4) + 2 grippers
      - Delta TCP (14):  L dpos(3) + L daa(3) + R dpos(3) + R daa(3) + L grip + R grip
    """

    print(f"=== Step {step} | Absolute TCP ===")
    # Absolute
    l_pos_a = action[0, 0:3]
    l_quat_a = action[0, 3:7]
    r_pos_a = action[0, 7:10]
    r_quat_a = action[0, 10:14]
    grips = action[0, 14:16]

    l_pos_o = state[0:3]
    l_quat_o = state[3:7]
    r_pos_o = state[7:10]
    r_quat_o = state[10:14]
    print(
        " Left  Pos   | Obs", _format_tensor(l_pos_o), " Pred", _format_tensor(l_pos_a)
    )
    print(
        " Left  Quat  | Obs",
        _format_tensor(l_quat_o),
        " Pred",
        _format_tensor(l_quat_a),
    )
    print(
        " Right Pos   | Obs", _format_tensor(r_pos_o), " Pred", _format_tensor(r_pos_a)
    )
    print(
        " Right Quat  | Obs",
        _format_tensor(r_quat_o),
        " Pred",
        _format_tensor(r_quat_a),
    )
    print(" Grippers (L,R)", _format_tensor(grips))


def print_tcp_delta(step: int, state: torch.Tensor, action: torch.Tensor):
    """
    Pretty print action vs (optional) observation.
    Formats:
      - Absolute TCP (16): L pos(3) + L quat(4) + R pos(3) + R quat(4) + 2 grippers
      - Delta TCP (14):  L dpos(3) + L daa(3) + R dpos(3) + R daa(3) + L grip + R grip
    """

    print(f"=== Step {step} | Delta TCP ===")
    # Delta TCP
    l_dpos = action[0, 0:3]
    l_daa = action[0, 3:6]
    r_dpos = action[0, 6:9]
    r_daa = action[0, 9:12]
    grips = action[0, 12:14]

    print(" Left  ΔPos  ", _format_tensor(l_dpos))
    print(" Left  ΔAA   ", _format_tensor(l_daa))
    print(" Right ΔPos  ", _format_tensor(r_dpos))
    print(" Right ΔAA   ", _format_tensor(r_daa))
    print(" Grippers (L,R)", _format_tensor(grips))

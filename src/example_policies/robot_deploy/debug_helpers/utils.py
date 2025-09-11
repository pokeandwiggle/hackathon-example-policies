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


def _fmt(t, precision: int = 3):
    if isinstance(t, torch.Tensor):
        t = t.detach().cpu()
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in t.tolist()) + "]"


def print_info(step: int, observation: dict, action: torch.Tensor):
    """
    Pretty print action vs (optional) observation.
    Formats:
      - Absolute TCP (16): L pos(3) + L quat(4) + R pos(3) + R quat(4) + 2 grippers
      - Delta TCP (14):  L dpos(3) + L daa(3) + R dpos(3) + R daa(3) + L grip + R grip
    """
    act_len = action.shape[-1]
    state = observation.get("observation.state")
    state0 = state[0] if isinstance(state, torch.Tensor) and state.ndim >= 2 else None

    print(f"=== Step {step} | Action shape {tuple(action.shape)} ===")

    if act_len == 16:
        # Absolute
        l_pos_a = action[0, 0:3]
        l_quat_a = action[0, 3:7]
        r_pos_a = action[0, 7:10]
        r_quat_a = action[0, 10:14]
        grips = action[0, 14:16]

        if state0 is not None and state0.shape[0] >= 16:
            l_pos_o = state0[0:3]
            l_quat_o = state0[3:7]
            r_pos_o = state0[7:10]
            r_quat_o = state0[10:14]
            print(" Left  Pos   | Obs", _fmt(l_pos_o), " Pred", _fmt(l_pos_a))
            print(" Left  Quat  | Obs", _fmt(l_quat_o), " Pred", _fmt(l_quat_a))
            print(" Right Pos   | Obs", _fmt(r_pos_o), " Pred", _fmt(r_pos_a))
            print(" Right Quat  | Obs", _fmt(r_quat_o), " Pred", _fmt(r_quat_a))
        else:
            print(" Left  Pos   ", _fmt(l_pos_a))
            print(" Left  Quat  ", _fmt(l_quat_a))
            print(" Right Pos   ", _fmt(r_pos_a))
            print(" Right Quat  ", _fmt(r_quat_a))
        print(" Grippers (L,R)", _fmt(grips))

    elif act_len == 14:
        # Delta TCP
        l_dpos = action[0, 0:3]
        l_daa = action[0, 3:6]
        r_dpos = action[0, 6:9]
        r_daa = action[0, 9:12]
        l_grip = action[0, 12]
        r_grip = action[0, 13]

        print(" Left  ΔPos  ", _fmt(l_dpos))
        print(" Left  ΔAA   ", _fmt(l_daa))
        print(" Right ΔPos  ", _fmt(r_dpos))
        print(" Right ΔAA   ", _fmt(r_daa))
        print(f" Grippers    [L={float(l_grip):.3f}, R={float(r_grip):.3f}]")

    else:
        print(f"Unsupported action length {act_len}; raw): {_fmt(action)}")

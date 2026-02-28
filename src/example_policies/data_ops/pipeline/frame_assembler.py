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

from ..config.pipeline_config import PipelineConfig
from ...utils.action_order import ActionMode
from .assembly.action_assembler import ActionAssembler
from .assembly.state_assembler import StateAssembler


class FrameAssembler:
    """Assembles the final frame from various components."""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.action_assembler = ActionAssembler(config)
        self.state_assembler = StateAssembler(config)
        self.last_abs_action = None
        self._pending_delta_frame = None

    def reset(self):
        self.action_assembler.reset()
        self.state_assembler.reset()

        self.last_abs_action = None
        self._pending_delta_frame = None

    def _uses_step_delta_actions(self) -> bool:
        return self.config.action_level in {
            ActionMode.DELTA_TCP,
            ActionMode.DELTA_JOINT,
        }

    def assemble(self, parsed_frame: dict) -> dict | None:
        """Assemble a frame.

        For delta action modes, returns a one-step delayed frame so that action at
        timestep ``t`` corresponds to transition ``t -> t+1``. The very first frame
        of an episode is buffered and returns ``None``.
        """
        frame = {}
        action_dict, abs_action = self.action_assembler.assemble(
            parsed_frame, self.last_abs_action
        )
        # if it is the very very first frame, set the last action to the pose
        #  at the start of the frame as computed by action translator
        if self.last_abs_action is None:
            self.last_abs_action = abs_action

        state_dict = self.state_assembler.assemble(parsed_frame, self.last_abs_action)
        img_dict = self.select_images(parsed_frame)
        frame.update(action_dict)
        frame.update(state_dict)
        frame.update(img_dict)

        self.last_abs_action = abs_action

        if not self._uses_step_delta_actions():
            return frame

        if self._pending_delta_frame is None:
            self._pending_delta_frame = frame
            return None

        # Align actions so frame[t].action is command from t -> t+1.
        self._pending_delta_frame["action"] = frame["action"]
        output = self._pending_delta_frame
        self._pending_delta_frame = frame
        return output

    def select_images(self, parsed_frame: dict):
        # Select observation.images.xxxxx keys
        images = {}

        for k, v in parsed_frame.items():
            if "image" in k:
                images[k] = v

        return images

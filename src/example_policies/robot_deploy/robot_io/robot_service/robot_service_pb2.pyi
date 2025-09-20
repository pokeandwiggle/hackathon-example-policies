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

from collections.abc import Iterable as _Iterable
from collections.abc import Mapping as _Mapping
from typing import ClassVar as _ClassVar
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper

DESCRIPTOR: _descriptor.FileDescriptor

class ExecutionMode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    EXECUTION_MODE_UNSPECIFIED: _ClassVar[ExecutionMode]
    EXECUTION_MODE_CARTESIAN_TARGET_QUEUE: _ClassVar[ExecutionMode]
    EXECUTION_MODE_CARTESIAN_TARGET: _ClassVar[ExecutionMode]
    EXECUTION_MODE_JOINT_TARGET: _ClassVar[ExecutionMode]
    EXECUTION_MODE_CARTESIAN_WAYPOINT: _ClassVar[ExecutionMode]

EXECUTION_MODE_UNSPECIFIED: ExecutionMode
EXECUTION_MODE_CARTESIAN_TARGET_QUEUE: ExecutionMode
EXECUTION_MODE_CARTESIAN_TARGET: ExecutionMode
EXECUTION_MODE_JOINT_TARGET: ExecutionMode
EXECUTION_MODE_CARTESIAN_WAYPOINT: ExecutionMode

class ResetDriversRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetDriversResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetRobotRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetRobotResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetVisionRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class ResetVisionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MoveHomeRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class MoveHomeResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RecoverErrorsRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class RecoverErrorsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class PrepareExecutionRequest(_message.Message):
    __slots__ = ("execution_mode",)
    EXECUTION_MODE_FIELD_NUMBER: _ClassVar[int]
    execution_mode: ExecutionMode
    def __init__(
        self, execution_mode: _Optional[_Union[ExecutionMode, str]] = ...
    ) -> None: ...

class PrepareExecutionResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class GetStateRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamStateRequest(_message.Message):
    __slots__ = ("target_hz",)
    TARGET_HZ_FIELD_NUMBER: _ClassVar[int]
    target_hz: float
    def __init__(self, target_hz: _Optional[float] = ...) -> None: ...

class GetStateResponse(_message.Message):
    __slots__ = ("current_state",)
    CURRENT_STATE_FIELD_NUMBER: _ClassVar[int]
    current_state: State
    def __init__(
        self, current_state: _Optional[_Union[State, _Mapping]] = ...
    ) -> None: ...

class EnqueueCartesianTargetsRequest(_message.Message):
    __slots__ = ("cartesian_targets",)
    CARTESIAN_TARGETS_FIELD_NUMBER: _ClassVar[int]
    cartesian_targets: _containers.RepeatedCompositeFieldContainer[CartesianTarget]
    def __init__(
        self,
        cartesian_targets: _Optional[
            _Iterable[_Union[CartesianTarget, _Mapping]]
        ] = ...,
    ) -> None: ...

class EnqueueCartesianTargetsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetCartesianTargetRequest(_message.Message):
    __slots__ = ("cartesian_target",)
    CARTESIAN_TARGET_FIELD_NUMBER: _ClassVar[int]
    cartesian_target: CartesianTarget
    def __init__(
        self, cartesian_target: _Optional[_Union[CartesianTarget, _Mapping]] = ...
    ) -> None: ...

class SetCartesianTargetResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetJointTargetRequest(_message.Message):
    __slots__ = ("joint_target",)
    JOINT_TARGET_FIELD_NUMBER: _ClassVar[int]
    joint_target: JointTarget
    def __init__(
        self, joint_target: _Optional[_Union[JointTarget, _Mapping]] = ...
    ) -> None: ...

class SetJointTargetResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class SetCartesianWaypointRequest(_message.Message):
    __slots__ = ("cartesian_waypoint",)
    CARTESIAN_WAYPOINT_FIELD_NUMBER: _ClassVar[int]
    cartesian_waypoint: CartesianTarget
    def __init__(
        self, cartesian_waypoint: _Optional[_Union[CartesianTarget, _Mapping]] = ...
    ) -> None: ...

class SetCartesianWaypointResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamJointTargetsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamCartesianTargetsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StreamCartesianWaypointsResponse(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CameraFrame(_message.Message):
    __slots__ = ("width", "height", "format", "data")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    width: int
    height: int
    format: str
    data: bytes
    def __init__(
        self,
        width: _Optional[int] = ...,
        height: _Optional[int] = ...,
        format: _Optional[str] = ...,
        data: _Optional[bytes] = ...,
    ) -> None: ...

class RobotState(_message.Message):
    __slots__ = ("pose", "velocity")
    POSE_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    pose: Pose
    velocity: Twist
    def __init__(
        self,
        pose: _Optional[_Union[Pose, _Mapping]] = ...,
        velocity: _Optional[_Union[Twist, _Mapping]] = ...,
    ) -> None: ...

class JointState(_message.Message):
    __slots__ = ("position", "velocity", "effort")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    EFFORT_FIELD_NUMBER: _ClassVar[int]
    position: float
    velocity: float
    effort: float
    def __init__(
        self,
        position: _Optional[float] = ...,
        velocity: _Optional[float] = ...,
        effort: _Optional[float] = ...,
    ) -> None: ...

class State(_message.Message):
    __slots__ = ("timestamp_ns", "cameras", "robots", "joints")

    class CamerasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: CameraFrame
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[CameraFrame, _Mapping]] = ...,
        ) -> None: ...

    class RobotsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: RobotState
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[RobotState, _Mapping]] = ...,
        ) -> None: ...

    class JointsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: JointState
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[JointState, _Mapping]] = ...,
        ) -> None: ...

    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    CAMERAS_FIELD_NUMBER: _ClassVar[int]
    ROBOTS_FIELD_NUMBER: _ClassVar[int]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ns: int
    cameras: _containers.MessageMap[str, CameraFrame]
    robots: _containers.MessageMap[str, RobotState]
    joints: _containers.MessageMap[str, JointState]
    def __init__(
        self,
        timestamp_ns: _Optional[int] = ...,
        cameras: _Optional[_Mapping[str, CameraFrame]] = ...,
        robots: _Optional[_Mapping[str, RobotState]] = ...,
        joints: _Optional[_Mapping[str, JointState]] = ...,
    ) -> None: ...

class Quaternion(_message.Message):
    __slots__ = ("x", "y", "z", "w")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    W_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    w: float
    def __init__(
        self,
        x: _Optional[float] = ...,
        y: _Optional[float] = ...,
        z: _Optional[float] = ...,
        w: _Optional[float] = ...,
    ) -> None: ...

class Vector3(_message.Message):
    __slots__ = ("x", "y", "z")
    X_FIELD_NUMBER: _ClassVar[int]
    Y_FIELD_NUMBER: _ClassVar[int]
    Z_FIELD_NUMBER: _ClassVar[int]
    x: float
    y: float
    z: float
    def __init__(
        self,
        x: _Optional[float] = ...,
        y: _Optional[float] = ...,
        z: _Optional[float] = ...,
    ) -> None: ...

class Pose(_message.Message):
    __slots__ = ("position", "orientation")
    POSITION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    position: Vector3
    orientation: Quaternion
    def __init__(
        self,
        position: _Optional[_Union[Vector3, _Mapping]] = ...,
        orientation: _Optional[_Union[Quaternion, _Mapping]] = ...,
    ) -> None: ...

class Twist(_message.Message):
    __slots__ = ("linear", "angular")
    LINEAR_FIELD_NUMBER: _ClassVar[int]
    ANGULAR_FIELD_NUMBER: _ClassVar[int]
    linear: Vector3
    angular: Vector3
    def __init__(
        self,
        linear: _Optional[_Union[Vector3, _Mapping]] = ...,
        angular: _Optional[_Union[Vector3, _Mapping]] = ...,
    ) -> None: ...

class CartesianTarget(_message.Message):
    __slots__ = ("robot_poses", "gripper_widths", "robot_stiffness_factors")

    class RobotPosesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: Pose
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[Pose, _Mapping]] = ...,
        ) -> None: ...

    class GripperWidthsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class RobotStiffnessFactorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    ROBOT_POSES_FIELD_NUMBER: _ClassVar[int]
    GRIPPER_WIDTHS_FIELD_NUMBER: _ClassVar[int]
    ROBOT_STIFFNESS_FACTORS_FIELD_NUMBER: _ClassVar[int]
    robot_poses: _containers.MessageMap[str, Pose]
    gripper_widths: _containers.ScalarMap[str, float]
    robot_stiffness_factors: _containers.ScalarMap[str, float]
    def __init__(
        self,
        robot_poses: _Optional[_Mapping[str, Pose]] = ...,
        gripper_widths: _Optional[_Mapping[str, float]] = ...,
        robot_stiffness_factors: _Optional[_Mapping[str, float]] = ...,
    ) -> None: ...

class JointTarget(_message.Message):
    __slots__ = ("joint_angles", "gripper_widths", "robot_stiffness_factors")

    class JointAnglesEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class GripperWidthsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    class RobotStiffnessFactorsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: float
        def __init__(
            self, key: _Optional[str] = ..., value: _Optional[float] = ...
        ) -> None: ...

    JOINT_ANGLES_FIELD_NUMBER: _ClassVar[int]
    GRIPPER_WIDTHS_FIELD_NUMBER: _ClassVar[int]
    ROBOT_STIFFNESS_FACTORS_FIELD_NUMBER: _ClassVar[int]
    joint_angles: _containers.ScalarMap[str, float]
    gripper_widths: _containers.ScalarMap[str, float]
    robot_stiffness_factors: _containers.ScalarMap[str, float]
    def __init__(
        self,
        joint_angles: _Optional[_Mapping[str, float]] = ...,
        gripper_widths: _Optional[_Mapping[str, float]] = ...,
        robot_stiffness_factors: _Optional[_Mapping[str, float]] = ...,
    ) -> None: ...

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

DESCRIPTOR: _descriptor.FileDescriptor

class StreamSnapshotRequest(_message.Message):
    __slots__ = ("frequency_hz",)
    FREQUENCY_HZ_FIELD_NUMBER: _ClassVar[int]
    frequency_hz: float
    def __init__(self, frequency_hz: _Optional[float] = ...) -> None: ...

class GetSnapshotRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class CameraActual(_message.Message):
    __slots__ = ("frame_id", "format", "data")
    FRAME_ID_FIELD_NUMBER: _ClassVar[int]
    FORMAT_FIELD_NUMBER: _ClassVar[int]
    DATA_FIELD_NUMBER: _ClassVar[int]
    frame_id: str
    format: str
    data: bytes
    def __init__(
        self,
        frame_id: _Optional[str] = ...,
        format: _Optional[str] = ...,
        data: _Optional[bytes] = ...,
    ) -> None: ...

class RobotActual(_message.Message):
    __slots__ = ("pose", "velocity")
    POSE_FIELD_NUMBER: _ClassVar[int]
    VELOCITY_FIELD_NUMBER: _ClassVar[int]
    pose: Pose
    velocity: float
    def __init__(
        self,
        pose: _Optional[_Union[Pose, _Mapping]] = ...,
        velocity: _Optional[float] = ...,
    ) -> None: ...

class JointActual(_message.Message):
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

class SnapshotResponse(_message.Message):
    __slots__ = ("timestamp_ns", "rgb_cameras", "depth_cameras", "robots", "joints")

    class RgbCamerasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: CameraActual
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[CameraActual, _Mapping]] = ...,
        ) -> None: ...

    class DepthCamerasEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: CameraActual
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[CameraActual, _Mapping]] = ...,
        ) -> None: ...

    class RobotsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: RobotActual
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[RobotActual, _Mapping]] = ...,
        ) -> None: ...

    class JointsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: JointActual
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[JointActual, _Mapping]] = ...,
        ) -> None: ...

    TIMESTAMP_NS_FIELD_NUMBER: _ClassVar[int]
    RGB_CAMERAS_FIELD_NUMBER: _ClassVar[int]
    DEPTH_CAMERAS_FIELD_NUMBER: _ClassVar[int]
    ROBOTS_FIELD_NUMBER: _ClassVar[int]
    JOINTS_FIELD_NUMBER: _ClassVar[int]
    timestamp_ns: int
    rgb_cameras: _containers.MessageMap[str, CameraActual]
    depth_cameras: _containers.MessageMap[str, CameraActual]
    robots: _containers.MessageMap[str, RobotActual]
    joints: _containers.MessageMap[str, JointActual]
    def __init__(
        self,
        timestamp_ns: _Optional[int] = ...,
        rgb_cameras: _Optional[_Mapping[str, CameraActual]] = ...,
        depth_cameras: _Optional[_Mapping[str, CameraActual]] = ...,
        robots: _Optional[_Mapping[str, RobotActual]] = ...,
        joints: _Optional[_Mapping[str, JointActual]] = ...,
    ) -> None: ...

class Pose(_message.Message):
    __slots__ = ("position", "orientation")

    class Point(_message.Message):
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

    POSITION_FIELD_NUMBER: _ClassVar[int]
    ORIENTATION_FIELD_NUMBER: _ClassVar[int]
    position: Pose.Point
    orientation: Pose.Quaternion
    def __init__(
        self,
        position: _Optional[_Union[Pose.Point, _Mapping]] = ...,
        orientation: _Optional[_Union[Pose.Quaternion, _Mapping]] = ...,
    ) -> None: ...

class Gripper(_message.Message):
    __slots__ = ("width",)
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    width: float
    def __init__(self, width: _Optional[float] = ...) -> None: ...

class RobotDesired(_message.Message):
    __slots__ = ("pose", "gripper_width")
    POSE_FIELD_NUMBER: _ClassVar[int]
    GRIPPER_WIDTH_FIELD_NUMBER: _ClassVar[int]
    pose: Pose
    gripper_width: float
    def __init__(
        self,
        pose: _Optional[_Union[Pose, _Mapping]] = ...,
        gripper_width: _Optional[float] = ...,
    ) -> None: ...

class Target(_message.Message):
    __slots__ = ("robots",)

    class RobotsEntry(_message.Message):
        __slots__ = ("key", "value")
        KEY_FIELD_NUMBER: _ClassVar[int]
        VALUE_FIELD_NUMBER: _ClassVar[int]
        key: str
        value: RobotDesired
        def __init__(
            self,
            key: _Optional[str] = ...,
            value: _Optional[_Union[RobotDesired, _Mapping]] = ...,
        ) -> None: ...

    ROBOTS_FIELD_NUMBER: _ClassVar[int]
    robots: _containers.MessageMap[str, RobotDesired]
    def __init__(
        self, robots: _Optional[_Mapping[str, RobotDesired]] = ...
    ) -> None: ...

class SetTargetRequest(_message.Message):
    __slots__ = ("targets",)
    TARGETS_FIELD_NUMBER: _ClassVar[int]
    targets: _containers.RepeatedCompositeFieldContainer[Target]
    def __init__(
        self, targets: _Optional[_Iterable[_Union[Target, _Mapping]]] = ...
    ) -> None: ...

class SetTargetResponse(_message.Message):
    __slots__ = ("success", "message")
    SUCCESS_FIELD_NUMBER: _ClassVar[int]
    MESSAGE_FIELD_NUMBER: _ClassVar[int]
    success: bool
    message: str
    def __init__(self, success: bool = ..., message: _Optional[str] = ...) -> None: ...

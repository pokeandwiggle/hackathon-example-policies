from dataclasses import dataclass

import grpc
import numpy as np
import torch

from example_policies.robot_deploy.action_translator import ActionMode, ActionTranslator
from example_policies.robot_deploy.robot_io.robot_client import RobotClient
from example_policies.robot_deploy.robot_io.robot_interface import (
    RobotInterface,
    _build_cart_target,
)
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


@dataclass
class DummyConfig:
    metadata = None


def main(server: str, action: torch.Tensor):
    channel = grpc.insecure_channel(server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)
    cfg = DummyConfig()
    robot_interface = RobotInterface(stub, cfg)
    robot_interface.send_action(action, ActionMode.ABS_TCP)


if __name__ == "__main__":
    server = "localhost:50051"
    tcp_numpy = np.array(
        [
            -0.22568389773368835,
            0.745937705039978,
            0.3866482377052307,
            0.04157385602593422,
            0.954410970211029,
            0.09571849554777145,
            0.27965912222862244,
            0.2665419578552246,
            0.7410882711410522,
            0.3717521131038666,
            0.004377448465675116,
            -0.937037467956543,
            -0.05347206071019173,
            0.345083087682724,
            0,
            0,
        ]
    )
    tcp_torch = torch.tensor(tcp_numpy)[None, :]
    main(server, tcp_torch)

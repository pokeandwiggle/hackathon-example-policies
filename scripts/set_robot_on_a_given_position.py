import grpc
import numpy as np

from example_policies.robot_deploy.robot_io.robot_client import RobotClient
from example_policies.robot_deploy.robot_io.robot_interface import _build_joint_target
from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2,
    robot_service_pb2_grpc,
)


def main(server: str, joint_target_numpy: np.ndarray):
    channel = grpc.insecure_channel(server)
    stub = robot_service_pb2_grpc.RobotServiceStub(channel)
    robot_client = RobotClient(stub)
    joint_target = _build_joint_target(joint_target_numpy)
    robot_client.send_joint_direct_target(joint_target)


if __name__ == "__main__":
    server = "localhost:50051"
    joint_target_numpy = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    main(server, joint_target_numpy)

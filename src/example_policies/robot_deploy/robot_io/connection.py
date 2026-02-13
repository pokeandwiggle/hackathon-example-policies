import grpc

from example_policies.robot_deploy.robot_io.robot_service import (
    robot_service_pb2_grpc,
)


class RobotConnection:
    """Manages gRPC connection lifecycle."""

    def __init__(self, server_address: str):
        self.server_address = server_address
        self.channel = None
        self.stub = None

    def __enter__(self):
        """Connect to robot service."""
        self.channel = grpc.insecure_channel(self.server_address)
        self.stub = robot_service_pb2_grpc.RobotServiceStub(self.channel)
        return self.stub

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close connection."""
        if self.channel:
            self.channel.close()

import threading
import time
from typing import Any, Dict, List, Optional, Tuple

import rclpy
from geometry_msgs.msg import PoseStamped, TwistStamped
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy

# ROS2 message types
from sensor_msgs.msg import CompressedImage, JointState
from std_msgs.msg import Float32, Float64, Float64MultiArray

from ...data_ops.config.rosbag_topics import RosTopicEnum


class ROS2RobotClient(Node):
    """
    ROS2-based robot client that provides similar functionality to the gRPC robot client.
    Communicates with the robot using ROS2 topics instead of gRPC services.
    """

    CART_QUEUE = "cartesian_target_queue"
    CART_DIRECT = "cartesian_target"
    CART_WAYPOINT = "cartesian_waypoint"
    JOINT_DIRECT = "joint_target"

    def __init__(self, node_name: str = "robot_client_node"):
        super().__init__(node_name)

        self.control_mode = None
        self._current_state_cache = {}
        self._state_lock = threading.Lock()

        # Initialize subscriber attributes
        self.joint_left_sub = None
        self.joint_right_sub = None
        self.tcp_left_sub = None
        self.tcp_right_sub = None
        self.gripper_left_dist_sub = None
        self.gripper_right_dist_sub = None
        self.camera_left_rgb_sub = None
        self.camera_right_rgb_sub = None
        self.camera_static_rgb_sub = None

        # QoS profile for reliable communication
        self.reliable_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            durability=DurabilityPolicy.VOLATILE,
            depth=10,
        )

        # Initialize publishers for robot control
        self._init_publishers()

        # Initialize subscribers for robot state
        self._init_subscribers()

        # Give some time for connections to establish
        time.sleep(0.5)

    def _init_publishers(self):
        """Initialize all publishers for robot control commands."""
        # TCP/Cartesian control publishers
        self.tcp_left_pub = self.create_publisher(
            TwistStamped, RosTopicEnum.DES_TCP_LEFT.value, self.reliable_qos
        )
        self.tcp_right_pub = self.create_publisher(
            TwistStamped, RosTopicEnum.DES_TCP_RIGHT.value, self.reliable_qos
        )

        # Gripper control publishers
        self.gripper_left_pub = self.create_publisher(
            Float64, RosTopicEnum.DES_GRIPPER_LEFT.value, self.reliable_qos
        )
        self.gripper_right_pub = self.create_publisher(
            Float64, RosTopicEnum.DES_GRIPPER_RIGHT.value, self.reliable_qos
        )

    def _init_subscribers(self):
        """Initialize all subscribers for robot state information."""
        # Joint state subscribers - cache both grouped and individual joints
        self.joint_left_sub = self.create_subscription(
            JointState,
            RosTopicEnum.ACTUAL_JOINT_LEFT.value,
            lambda msg: self._update_joint_state_cache("left", msg),
            self.reliable_qos,
        )
        self.joint_right_sub = self.create_subscription(
            JointState,
            RosTopicEnum.ACTUAL_JOINT_RIGHT.value,
            lambda msg: self._update_joint_state_cache("right", msg),
            self.reliable_qos,
        )

        # TCP/Pose subscribers
        self.tcp_left_sub = self.create_subscription(
            PoseStamped,
            RosTopicEnum.ACTUAL_TCP_LEFT.value,
            lambda msg: self._update_state_cache("tcp_left", msg),
            self.reliable_qos,
        )
        self.tcp_right_sub = self.create_subscription(
            PoseStamped,
            RosTopicEnum.ACTUAL_TCP_RIGHT.value,
            lambda msg: self._update_state_cache("tcp_right", msg),
            self.reliable_qos,
        )

        # Gripper state subscribers (finger_distance_mm topics use Float32)
        self.gripper_left_dist_sub = self.create_subscription(
            Float32,
            RosTopicEnum.LEFT_GRIPPER_DIST.value,
            lambda msg: self._update_state_cache("gripper_left_dist", msg),
            self.reliable_qos,
        )
        self.gripper_right_dist_sub = self.create_subscription(
            Float32,
            RosTopicEnum.RIGHT_GRIPPER_DIST.value,
            lambda msg: self._update_state_cache("gripper_right_dist", msg),
            self.reliable_qos,
        )

        # Camera subscribers
        self._init_camera_subscribers()

    def _init_camera_subscribers(self):
        """Initialize camera subscribers for all available camera topics."""
        # Wrist cameras
        self.camera_left_rgb_sub = self.create_subscription(
            CompressedImage,
            RosTopicEnum.RGB_LEFT_IMAGE.value,
            lambda msg: self._update_camera_cache("cam_left_color_optical_frame", msg),
            self.reliable_qos,
        )
        self.camera_right_rgb_sub = self.create_subscription(
            CompressedImage,
            RosTopicEnum.RGB_RIGHT_IMAGE.value,
            lambda msg: self._update_camera_cache("cam_right_color_optical_frame", msg),
            self.reliable_qos,
        )

        # Static camera
        self.camera_static_rgb_sub = self.create_subscription(
            CompressedImage,
            RosTopicEnum.RGB_STATIC_IMAGE.value,
            lambda msg: self._update_camera_cache("cam_static_optical_frame", msg),
            self.reliable_qos,
        )

    def _update_state_cache(self, key: str, msg: Any):
        """Update the internal state cache with new sensor data."""
        with self._state_lock:
            self._current_state_cache[key] = {
                "data": msg,
                "timestamp": self.get_clock().now(),
            }

    def _update_joint_state_cache(self, robot_side: str, msg: JointState):
        """Update joint state cache for both grouped and individual joint access."""
        with self._state_lock:
            # Store grouped joint state
            self._current_state_cache[f"joint_{robot_side}"] = {
                "data": msg,
                "timestamp": self.get_clock().now(),
            }

            # Store individual joints for observation builder access
            if "joints" not in self._current_state_cache:
                self._current_state_cache["joints"] = {}

            for i, joint_name in enumerate(msg.name):
                # Create joint object with position, velocity, effort attributes
                class JointData:
                    def __init__(self, pos, vel, eff):
                        self.position = pos
                        self.velocity = vel
                        self.effort = eff

                joint_obj = JointData(
                    msg.position[i] if i < len(msg.position) else 0.0,
                    msg.velocity[i] if i < len(msg.velocity) else 0.0,
                    msg.effort[i] if i < len(msg.effort) else 0.0,
                )
                self._current_state_cache["joints"][joint_name] = joint_obj

    def _update_camera_cache(self, camera_name: str, msg: CompressedImage):
        """Update camera cache for observation builder access."""
        with self._state_lock:
            if "cameras" not in self._current_state_cache:
                self._current_state_cache["cameras"] = {}

            # Create camera data object that mimics the expected structure
            class CameraData:
                def __init__(self, data):
                    self.data = data

            self._current_state_cache["cameras"][camera_name] = CameraData(msg.data)

    def get_snapshot(self) -> Tuple[Optional[Any], Optional[List[str]]]:
        """
        Get a snapshot of the current robot state.
        Returns data structure compatible with observation_builder.py

        Returns:
            Tuple of (state_object, robot_names) similar to the gRPC client
        """
        with self._state_lock:
            if not self._current_state_cache:
                self.get_logger().warn("No robot state data available yet")
                return None, None

            # Create a state object that mimics the gRPC snapshot structure
            # This allows the observation_builder to work with minimal changes
            class SnapshotState:
                def __init__(self, cache):
                    self.robots = self._create_robot_dict(cache)
                    self.joints = cache.get("joints", {})
                    self.cameras = cache.get("cameras", {})
                    # Add gripper distances (Float32 values from finger_distance_mm topics)
                    self.gripper_left_dist = self._get_gripper_distance(
                        cache, "gripper_left_dist"
                    )
                    self.gripper_right_dist = self._get_gripper_distance(
                        cache, "gripper_right_dist"
                    )

                def _get_gripper_distance(self, cache, key):
                    """Extract gripper distance value from cached Float32 message."""
                    gripper_data = cache.get(key)
                    if gripper_data and "data" in gripper_data:
                        msg = gripper_data["data"]
                        # Handle Float32 message from finger_distance_mm topic
                        if hasattr(msg, "data"):
                            return float(msg.data)
                    return 0.0

                def _create_robot_dict(self, cache):
                    robots = {}
                    for side in ["left", "right"]:
                        tcp_data = cache.get(f"tcp_{side}")
                        if tcp_data:
                            # Create a pose object that mimics the gRPC structure
                            class PoseObject:
                                def __init__(self, pose_msg):
                                    self.position = pose_msg["data"].pose.position
                                    self.orientation = pose_msg["data"].pose.orientation

                            robots[side] = type(
                                "Robot", (), {"pose": PoseObject(tcp_data)}
                            )()
                    return robots

            snapshot_state = SnapshotState(self._current_state_cache)
            robot_names = ["left", "right"]
            return snapshot_state, robot_names

    def send_cart_queue_target(self, cart_target: Dict) -> bool:
        """
        Send cartesian target using queue mode.

        Args:
            cart_target: Dictionary containing cartesian target information
            Expected keys: 'robot_name', 'position', 'orientation', 'gripper'
        """
        ctrl_mode = self.CART_QUEUE

        if self.control_mode != ctrl_mode:
            self.get_logger().info(f"Switching to control mode: {ctrl_mode}")
            self.control_mode = ctrl_mode

        return self._send_cartesian_command(cart_target)

    def send_cart_waypoint(self, cart_target: Dict) -> bool:
        """
        Send cartesian waypoint target.

        Args:
            cart_target: Dictionary containing cartesian target information
        """
        ctrl_mode = self.CART_WAYPOINT

        if self.control_mode != ctrl_mode:
            self.get_logger().info(f"Switching to control mode: {ctrl_mode}")
            self.control_mode = ctrl_mode

        return self._send_cartesian_command(cart_target)

    def send_cart_direct_target(self, cart_target: Dict) -> bool:
        """
        Send direct cartesian target.

        Args:
            cart_target: Dictionary containing cartesian target information
        """
        ctrl_mode = self.CART_DIRECT

        if self.control_mode != ctrl_mode:
            self.get_logger().info(f"Switching to control mode: {ctrl_mode}")
            self.control_mode = ctrl_mode

        return self._send_cartesian_command(cart_target)

    def _send_cartesian_command(self, cart_target: Dict) -> bool:
        """
        Internal method to send cartesian commands via ROS2 topics.

        Args:
            cart_target: Dictionary with keys 'robot_name', 'twist', 'gripper'
        """
        try:
            robot_name = cart_target.get("robot_name", "left")
            twist_data = cart_target.get("twist", {})
            gripper_value = cart_target.get("gripper", 0.0)

            # Create TwistStamped message
            twist_msg = TwistStamped()
            twist_msg.header.stamp = self.get_clock().now().to_msg()
            twist_msg.header.frame_id = f"{robot_name}_tcp"

            # Set linear velocity
            linear = twist_data.get("linear", {})
            twist_msg.twist.linear.x = linear.get("x", 0.0)
            twist_msg.twist.linear.y = linear.get("y", 0.0)
            twist_msg.twist.linear.z = linear.get("z", 0.0)

            # Set angular velocity
            angular = twist_data.get("angular", {})
            twist_msg.twist.angular.x = angular.get("x", 0.0)
            twist_msg.twist.angular.y = angular.get("y", 0.0)
            twist_msg.twist.angular.z = angular.get("z", 0.0)

            # Publish TCP command
            if robot_name == "left":
                self.tcp_left_pub.publish(twist_msg)
            elif robot_name == "right":
                self.tcp_right_pub.publish(twist_msg)
            else:
                self.get_logger().error(f"Unknown robot name: {robot_name}")
                return False

            # Publish gripper command
            gripper_msg = Float64()
            gripper_msg.data = float(gripper_value)

            if robot_name == "left":
                self.gripper_left_pub.publish(gripper_msg)
            elif robot_name == "right":
                self.gripper_right_pub.publish(gripper_msg)

            return True

        except (ValueError, AttributeError, KeyError) as e:
            self.get_logger().error(f"Failed to send cartesian command: {str(e)}")
            return False

    def send_joint_direct_target(self, joint_target: Dict) -> bool:  # noqa: ARG002
        """
        Send direct joint target.
        Note: This implementation uses twist commands as the current ROS topics
        don't expose direct joint control. This would need adaptation based on
        available joint control topics.

        Args:
            joint_target: Dictionary containing joint target information (unused)
        """
        ctrl_mode = self.JOINT_DIRECT

        if self.control_mode != ctrl_mode:
            self.get_logger().info(f"Switching to control mode: {ctrl_mode}")
            self.control_mode = ctrl_mode

        # For now, log that joint control is not fully implemented
        # This would require additional ROS topics for direct joint control
        self.get_logger().warn(
            "Direct joint control not fully implemented. "
            "Current ROS topics primarily support TCP control."
        )

        return False

    def send_move_home(self) -> bool:
        """
        Send a request to move the robot to its home position.
        This implementation sends zero velocities to stop the robot.
        A proper home position would require a dedicated service or action.

        Returns:
            True if the command was sent successfully
        """
        # Reset control_mode
        self.control_mode = None

        try:
            # Send zero velocities to both robots to stop them
            zero_twist = TwistStamped()
            zero_twist.header.stamp = self.get_clock().now().to_msg()

            # Stop left robot
            zero_twist.header.frame_id = "left_tcp"
            self.tcp_left_pub.publish(zero_twist)

            # Stop right robot
            zero_twist.header.frame_id = "right_tcp"
            self.tcp_right_pub.publish(zero_twist)

            self.get_logger().info("Sent move home command (stop velocities)")
            return True

        except (ValueError, AttributeError) as e:
            self.get_logger().error(f"Failed to send move home command: {str(e)}")
            return False

    def shutdown(self):
        """Clean shutdown of the ROS2 client."""
        self.get_logger().info("Shutting down ROS2 robot client")
        self.destroy_node()


# Convenience function to create and manage the ROS2 client
def create_ros2_robot_client(node_name: str = "robot_client_node") -> ROS2RobotClient:
    """
    Create a ROS2 robot client instance.

    Args:
        node_name: Name for the ROS2 node

    Returns:
        ROS2RobotClient instance
    """
    if not rclpy.ok():
        rclpy.init()

    return ROS2RobotClient(node_name)

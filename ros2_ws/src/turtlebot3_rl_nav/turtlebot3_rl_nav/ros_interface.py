from __future__ import annotations

import math
import threading
import time
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import LaserScan


class ROSInterface(Node):
    """
    Low-level ROS 2 interface for the TurtleBot3.

    Responsibilities:
        - Subscribe to /scan
        - Subscribe to /odom
        - Publish TwistStamped commands to /cmd_vel
        - Reset the robot pose through Gazebo

    This class contains no Gymnasium or RL logic.
    """

    WORLD_NAME = "default"
    ROBOT_NAME = "burger"
    ROBOT_TYPE = Entity.MODEL

    def __init__(self) -> None:
        super().__init__("turtlebot3_rl_nav")

        # --------------------------------------------------------------
        # Latest sensor data
        # --------------------------------------------------------------

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None

        self._scan_lock = threading.Lock()
        self._odom_lock = threading.Lock()

        # --------------------------------------------------------------
        # ROS subscriptions
        # --------------------------------------------------------------

        self.scan_subscription = self.create_subscription(
            LaserScan,
            "/scan",
            self._scan_callback,
            10,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/odom",
            self._odom_callback,
            10,
        )

        # --------------------------------------------------------------
        # ROS publisher
        # --------------------------------------------------------------

        self.cmd_vel_publisher = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10,
        )

        # --------------------------------------------------------------
        # Gazebo pose-reset service
        # --------------------------------------------------------------

        self.set_pose_client = self.create_client(
            SetEntityPose,
            f"/world/{self.WORLD_NAME}/set_pose",
        )

        self.get_logger().info(
            f"ROS interface initialized for world '{self.WORLD_NAME}'."
        )

    # ==================================================================
    # Sensor callbacks
    # ==================================================================

    def _scan_callback(self, msg: LaserScan) -> None:
        """Store the newest LiDAR message."""
        with self._scan_lock:
            self.latest_scan = msg

    def _odom_callback(self, msg: Odometry) -> None:
        """Store the newest odometry message."""
        with self._odom_lock:
            self.latest_odom = msg

    # ==================================================================
    # Sensor access
    # ==================================================================

    def get_latest_scan(self) -> Optional[LaserScan]:
        """Return the most recently received LiDAR message."""
        with self._scan_lock:
            return self.latest_scan

    def get_latest_odom(self) -> Optional[Odometry]:
        """Return the most recently received odometry message."""
        with self._odom_lock:
            return self.latest_odom

    # ==================================================================
    # Velocity control
    # ==================================================================

    def publish_velocity(
        self,
        linear_x: float,
        angular_z: float,
    ) -> None:
        """
        Publish a velocity command to /cmd_vel.

        Args:
            linear_x: Linear velocity in m/s.
            angular_z: Angular velocity in rad/s.
        """
        msg = TwistStamped()

        msg.header.stamp = self.get_clock().now().to_msg()

        msg.twist.linear.x = float(linear_x)
        msg.twist.linear.y = 0.0
        msg.twist.linear.z = 0.0

        msg.twist.angular.x = 0.0
        msg.twist.angular.y = 0.0
        msg.twist.angular.z = float(angular_z)

        self.cmd_vel_publisher.publish(msg)

    def stop_robot(self) -> None:
        """Publish a zero-velocity command."""
        self.publish_velocity(0.0, 0.0)

    # ==================================================================
    # Gazebo reset
    # ==================================================================

    def reset_robot_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        z: float = 0.01,
        timeout: float = 2.0,
    ) -> None:
        """Reset the Burger pose using Gazebo's SetEntityPose service."""

        service_name = f"/world/{self.WORLD_NAME}/set_pose"

        if not self.set_pose_client.wait_for_service(
            timeout_sec=timeout
        ):
            raise RuntimeError(
                f"Gazebo set-pose service is unavailable: "
                f"{service_name}"
            )

        request = SetEntityPose.Request()

        request.entity.name = self.ROBOT_NAME
        request.entity.type = self.ROBOT_TYPE

        request.pose.position.x = float(x)
        request.pose.position.y = float(y)
        request.pose.position.z = float(z)

        request.pose.orientation.x = 0.0
        request.pose.orientation.y = 0.0
        request.pose.orientation.z = math.sin(yaw / 2.0)
        request.pose.orientation.w = math.cos(yaw / 2.0)

        future = self.set_pose_client.call_async(request)

        start_time = time.monotonic()

        while not future.done():
            if time.monotonic() - start_time >= timeout:
                raise TimeoutError(
                    "Timed out waiting for Gazebo set-pose response."
                )

            time.sleep(0.01)

        if future.exception() is not None:
            raise RuntimeError(
                f"Gazebo set-pose service failed: "
                f"{future.exception()}"
            )

        response = future.result()

        if response is None:
            raise RuntimeError(
                "Gazebo returned an empty set-pose response."
            )

        if not response.success:
            raise RuntimeError(
                f"Gazebo rejected pose reset for entity "
                f"'{self.ROBOT_NAME}'."
            )


    def clear_sensor_data(self) -> None:
        """Clear cached sensor messages so the next reads are fresh."""
        with self._scan_lock:
            self.latest_scan = None

        with self._odom_lock:
            self.latest_odom = None
    # ==================================================================
    # Lifecycle
    # ==================================================================

    def destroy(self) -> None:
        """Stop the robot before shutting down."""
        self.stop_robot()
        self.destroy_node()


class ROSInterfaceRunner:
    """
    Runs ROSInterface in a background executor thread.

    This allows TurtleBotEnv to interact with ROS without manually
    spinning the ROS executor inside every Gymnasium operation.
    """

    def __init__(self) -> None:
        if not rclpy.ok():
            rclpy.init()

        self.node = ROSInterface()

        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )

        self._thread.start()

    def reset_robot_pose(
        self,
        x: float,
        y: float,
        yaw: float,
        z: float = 0.01,
        timeout: float = 2.0,
    ) -> None:
        """Reset the robot pose through the ROS interface."""
        self.node.reset_robot_pose(
            x=x,
            y=y,
            yaw=yaw,
            z=z,
            timeout=timeout,
        )

    def shutdown(self) -> None:
        """Stop ROS execution and clean up resources."""
        self.node.stop_robot()

        self.executor.shutdown()

        if self._thread.is_alive():
            self._thread.join(timeout=1.0)

        self.node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

        

    

    def clear_sensor_data(self) -> None:
        """Clear cached sensor messages."""
        self.node.clear_sensor_data()

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
from rclpy.signals import SignalHandlerOptions
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose
from sensor_msgs.msg import LaserScan


def stamp_to_ns(stamp) -> int:
    """Convert a builtin_interfaces/Time stamp to integer nanoseconds."""
    return stamp.sec * 1_000_000_000 + stamp.nanosec


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

        # Shared by both sensor callbacks so waiters are woken on
        # every new message.
        self._sensor_cv = threading.Condition()

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
        """Store the newest LiDAR message and wake any waiters."""
        with self._sensor_cv:
            self.latest_scan = msg
            self._sensor_cv.notify_all()

    def _odom_callback(self, msg: Odometry) -> None:
        """Store the newest odometry message and wake any waiters."""
        with self._sensor_cv:
            self.latest_odom = msg
            self._sensor_cv.notify_all()

    # ==================================================================
    # Sensor access
    # ==================================================================

    def get_latest_scan(self) -> Optional[LaserScan]:
        """Return the most recently received LiDAR message."""
        with self._sensor_cv:
            return self.latest_scan

    def get_latest_odom(self) -> Optional[Odometry]:
        """Return the most recently received odometry message."""
        with self._sensor_cv:
            return self.latest_odom

    def latest_odom_stamp_ns(self) -> int:
        """
        Return the newest odometry stamp in nanoseconds of sim time.

        Odometry arrives at 30 Hz, so this is a close lower bound on
        the current simulation time. Returns 0 before any odometry.
        """
        with self._sensor_cv:
            if self.latest_odom is None:
                return 0

            return stamp_to_ns(self.latest_odom.header.stamp)

    def wait_for_sensor_data(
        self,
        min_stamp_ns: int,
        timeout: float,
    ) -> tuple[LaserScan, Odometry]:
        """
        Block until both /scan and /odom have a message stamped at or
        after min_stamp_ns, then return that pair.

        Args:
            min_stamp_ns: Earliest acceptable header stamp (sim time).
            timeout: Wall-clock seconds before giving up.
        """

        def is_fresh() -> bool:
            return (
                self.latest_scan is not None
                and self.latest_odom is not None
                and stamp_to_ns(self.latest_scan.header.stamp)
                >= min_stamp_ns
                and stamp_to_ns(self.latest_odom.header.stamp)
                >= min_stamp_ns
            )

        with self._sensor_cv:
            if not self._sensor_cv.wait_for(is_fresh, timeout=timeout):
                scan_ns = (
                    stamp_to_ns(self.latest_scan.header.stamp)
                    if self.latest_scan is not None
                    else None
                )
                odom_ns = (
                    stamp_to_ns(self.latest_odom.header.stamp)
                    if self.latest_odom is not None
                    else None
                )

                raise TimeoutError(
                    f"Timed out after {timeout:.1f} s waiting for "
                    f"/scan and /odom stamped >= {min_stamp_ns} ns. "
                    f"Latest /scan: {scan_ns}, latest /odom: {odom_ns}."
                )

            return self.latest_scan, self.latest_odom

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
        """Publish a zero-velocity command if ROS is still running."""
        if not self.context.ok():
            return

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
            # Leave Ctrl-C to Python so the training script can save a
            # checkpoint and stop the robot before ROS shuts down.
            rclpy.init(signal_handler_options=SignalHandlerOptions.NO)

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

        # Destroying the node while the spin thread still runs (an
        # overloaded system can outlast the join) raises InvalidHandle in
        # that thread and buries the real error. The process is exiting
        # either way, so leave the node to it.
        if not self._thread.is_alive():
            self.node.destroy_node()

        if rclpy.ok():
            rclpy.shutdown()
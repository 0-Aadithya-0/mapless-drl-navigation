import threading
from typing import Optional

import rclpy
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ROSInterface(Node):
    """
    Low-level ROS 2 interface for the TurtleBot3.

    Responsibilities:
        - Subscribe to /scan
        - Subscribe to /odom
        - Publish TwistStamped commands to /cmd_vel

    """

    def __init__(self) -> None:
        super().__init__("turtlebot3_rl_nav")

        self.latest_scan: Optional[LaserScan] = None
        self.latest_odom: Optional[Odometry] = None

        self._scan_lock = threading.Lock()
        self._odom_lock = threading.Lock()

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

        self.cmd_vel_publisher = self.create_publisher(
            TwistStamped,
            "/cmd_vel",
            10,
        )

        self.get_logger().info("ROS interface initialized.")

    def _scan_callback(self, msg: LaserScan) -> None:
        """Store the newest LiDAR message."""
        with self._scan_lock:
            self.latest_scan = msg

    def _odom_callback(self, msg: Odometry) -> None:
        """Store the newest odometry message."""
        with self._odom_lock:
            self.latest_odom = msg

    def get_latest_scan(self) -> Optional[LaserScan]:
        """Return the most recently received LiDAR message."""
        with self._scan_lock:
            return self.latest_scan

    def get_latest_odom(self) -> Optional[Odometry]:
        """Return the most recently received odometry message."""
        with self._odom_lock:
            return self.latest_odom

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


class ROSInterfaceRunner:
    """
    Runs ROSInterface in a background executor thread.

    This keeps ROS spinning independently from the code that will
    eventually implement TurtleBotEnv.
    """

    def __init__(self) -> None:
        self.node = ROSInterface()
        self.executor = SingleThreadedExecutor()
        self.executor.add_node(self.node)

        self._thread = threading.Thread(
            target=self.executor.spin,
            daemon=True,
        )
        self._thread.start()

    def shutdown(self) -> None:
        """Stop ROS execution and clean up resources."""
        self.node.stop_robot()

        self.executor.shutdown()
        self.node.destroy_node()

        if self._thread.is_alive():
            self._thread.join(timeout=1.0)
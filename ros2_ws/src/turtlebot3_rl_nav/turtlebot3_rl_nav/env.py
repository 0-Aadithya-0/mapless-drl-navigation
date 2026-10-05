from __future__ import annotations

import time
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .lidar_processing import (
    LIDAR_MAX_RANGE,
    NUM_BEAMS,
    process_lidar_scan,
)
from .ros_interface import ROSInterfaceRunner


class TurtleBotEnv(gym.Env):
    """
    Gymnasium environment for TurtleBot3 obstacle avoidance.

    Week 2 DQN environment:

        Observation:
            24 normalized LiDAR beams in [0, 1]

        Actions:
            0 = forward
            1 = left
            2 = right
            3 = stop

        Termination:
            collision

        Truncation:
            maximum episode length

    ROS communication is handled entirely by ROSInterfaceRunner.
    """

    metadata = {"render_modes": []}

    # ==================================================================
    # Action definitions
    # ==================================================================

    ACTIONS = {
        0: (0.15, 0.0),   # forward
        1: (0.0, 0.8),    # left
        2: (0.0, -0.8),   # right
        3: (0.0, 0.0),    # stop
    }

    # ==================================================================
    # Environment configuration
    # ==================================================================

    CONTROL_DT = 0.20

    COLLISION_DISTANCE = 0.18
    COLLISION_REWARD = -10.0

    MAX_EPISODE_STEPS = 500

    # Stage 2 center is a safe starting point.
    START_X = 0.0
    START_Y = 0.0
    START_YAW = 0.0
    START_Z = 0.01

    SENSOR_TIMEOUT = 2.0

    # ==================================================================
    # Initialization
    # ==================================================================

    def __init__(self, max_episode_steps: int | None = None) -> None:
        super().__init__()

        # 24 normalized LiDAR values.
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_BEAMS,),
            dtype=np.float32,
        )

        # 4 discrete actions.
        self.action_space = spaces.Discrete(4)

        self.ros = ROSInterfaceRunner()

        self.current_step = 0
        if max_episode_steps is None:
            self.max_episode_steps = self.MAX_EPISODE_STEPS
        else:
            if max_episode_steps <= 0:
                raise ValueError(
                    "max_episode_steps must be greater than zero."
                )

        self.max_episode_steps = int(max_episode_steps)

    # ==================================================================
    # Gymnasium reset
    # ==================================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and return the initial observation.
        """
        super().reset(seed=seed)

        self.current_step = 0

        # Make sure the robot is not carrying velocity from
        # the previous episode.
        self.ros.node.stop_robot()

        # Discard old sensor data.
        self.ros.clear_sensor_data()

        # Teleport Burger to the starting pose.
        self.ros.reset_robot_pose(
            x=self.START_X,
            y=self.START_Y,
            yaw=self.START_YAW,
            z=self.START_Z,
            timeout=self.SENSOR_TIMEOUT,
        )

        # Stop again after teleportation.
        self.ros.node.stop_robot()

        # Wait for new /scan and /odom messages.
        self._wait_for_sensor_data()

        scan = self.ros.node.get_latest_scan()

        if scan is None:
            raise RuntimeError(
                "No LiDAR data received after environment reset."
            )

        observation = process_lidar_scan(scan)

        self._validate_observation(observation)

        info = {
            "step": 0,
            "collision": False,
            "min_lidar_distance": self._minimum_lidar_distance(scan),
        }

        return observation, info

    # ==================================================================
    # Gymnasium step
    # ==================================================================

    def step(
        self,
        action: int,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        """
        Execute one discrete action.
        """

        if not self.action_space.contains(action):
            raise ValueError(
                f"Invalid action {action}. "
                f"Expected an integer in [0, 3]."
            )

        action = int(action)

        linear_velocity, angular_velocity = self.ACTIONS[action]

        # We want the sensor data produced by this action,
        # not a previously cached message.
        self.ros.clear_sensor_data()

        # Send action to the robot.
        self.ros.node.publish_velocity(
            linear_x=linear_velocity,
            angular_z=angular_velocity,
        )

        # Hold the action for the control interval.
        time.sleep(self.CONTROL_DT)

        # Make sure fresh data has arrived.
        self._wait_for_sensor_data()

        scan = self.ros.node.get_latest_scan()
        odom = self.ros.node.get_latest_odom()

        if scan is None:
            raise RuntimeError(
                "No LiDAR data received during environment step."
            )

        if odom is None:
            raise RuntimeError(
                "No odometry data received during environment step."
            )

        # --------------------------------------------------------------
        # Observation
        # --------------------------------------------------------------

        observation = process_lidar_scan(scan)

        self._validate_observation(observation)

        # --------------------------------------------------------------
        # Collision detection
        # --------------------------------------------------------------

        min_distance = self._minimum_lidar_distance(scan)

        collision = (
            min_distance <= self.COLLISION_DISTANCE
        )

        # --------------------------------------------------------------
        # Reward
        # --------------------------------------------------------------

        forward_velocity = float(
            odom.twist.twist.linear.x
        )

        # Forward displacement during this control interval.
        reward = forward_velocity * self.CONTROL_DT

        if collision:
            reward += self.COLLISION_REWARD

        # --------------------------------------------------------------
        # Episode state
        # --------------------------------------------------------------

        self.current_step += 1

        terminated = collision

        truncated = (
            self.current_step >= self.max_episode_steps
            and not terminated
        )

        # Always stop after a terminal collision.
        if terminated:
            self.ros.node.stop_robot()

        info = {
            "step": self.current_step,
            "collision": collision,
            "min_lidar_distance": min_distance,
            "linear_velocity": forward_velocity,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # Sensor synchronization
    # ==================================================================

    def _wait_for_sensor_data(
        self,
        timeout: float | None = None,
    ) -> None:
        """
        Wait until both LiDAR and odometry data are available.
        """

        if timeout is None:
            timeout = self.SENSOR_TIMEOUT

        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:

            scan = self.ros.node.get_latest_scan()
            odom = self.ros.node.get_latest_odom()

            if scan is not None and odom is not None:
                return

            time.sleep(0.01)

        raise TimeoutError(
            "Timed out waiting for /scan and /odom data."
        )

    # ==================================================================
    # LiDAR processing for collision detection
    # ==================================================================

    @staticmethod
    def _minimum_lidar_distance(
        scan: Any,
    ) -> float:
        """
        Return the minimum valid LiDAR distance in meters.
        """

        ranges = np.asarray(
            scan.ranges,
            dtype=np.float32,
        )

        if ranges.size == 0:
            return LIDAR_MAX_RANGE

        # No-return values are not collisions.
        ranges = np.nan_to_num(
            ranges,
            nan=LIDAR_MAX_RANGE,
            posinf=LIDAR_MAX_RANGE,
            neginf=LIDAR_MAX_RANGE,
        )

        ranges = np.clip(
            ranges,
            0.0,
            LIDAR_MAX_RANGE,
        )

        return float(np.min(ranges))

    # ==================================================================
    # Observation validation
    # ==================================================================

    def _validate_observation(
        self,
        observation: np.ndarray,
    ) -> None:
        """Validate the fixed Week 2 observation contract."""

        if observation.shape != (NUM_BEAMS,):
            raise RuntimeError(
                f"Invalid observation shape: "
                f"{observation.shape}. "
                f"Expected ({NUM_BEAMS},)."
            )

        if observation.dtype != np.float32:
            raise RuntimeError(
                f"Invalid observation dtype: "
                f"{observation.dtype}. "
                "Expected float32."
            )

        if not np.all(np.isfinite(observation)):
            raise RuntimeError(
                "Observation contains NaN or infinite values."
            )

        if np.any(observation < 0.0) or np.any(
            observation > 1.0
        ):
            raise RuntimeError(
                "Observation contains values outside [0, 1]."
            )

    # ==================================================================
    # Cleanup
    # ==================================================================

    def close(self) -> None:
        """Stop the robot and shut down ROS."""

        self.ros.node.stop_robot()
        self.ros.shutdown()
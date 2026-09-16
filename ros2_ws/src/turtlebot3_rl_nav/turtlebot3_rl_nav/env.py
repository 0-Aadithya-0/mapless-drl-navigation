from __future__ import annotations

import math
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
    Gymnasium environment for TurtleBot3 obstacle avoidance in Gazebo.

    Observation:
        24 normalized LiDAR beams.

    Actions:
        0 -> forward
        1 -> left
        2 -> right
        3 -> stop

    Reward:
        Positive for forward motion.
        Negative for collision.

    Termination:
        Collision -> terminated=True
        Maximum episode length -> truncated=True
    """

    metadata = {"render_modes": []}

    # ------------------------------------------------------------------
    # Action parameters
    # ------------------------------------------------------------------

    FORWARD_LINEAR_VELOCITY = 0.15
    TURN_ANGULAR_VELOCITY = 0.8

    ACTIONS = {
        0: (FORWARD_LINEAR_VELOCITY, 0.0),   # forward
        1: (0.0, TURN_ANGULAR_VELOCITY),     # left
        2: (0.0, -TURN_ANGULAR_VELOCITY),   # right
        3: (0.0, 0.0),                       # stop
    }

    # ------------------------------------------------------------------
    # Environment parameters
    # ------------------------------------------------------------------

    CONTROL_DT = 0.20

    COLLISION_DISTANCE = 0.18

    MAX_EPISODE_STEPS = 500

    START_X = 0.0
    START_Y = 0.0
    START_YAW = 0.0

    COLLISION_REWARD = -10.0

    def __init__(self) -> None:
        super().__init__()

        # --------------------------------------------------------------
        # Gymnasium spaces
        # --------------------------------------------------------------

        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(NUM_BEAMS,),
            dtype=np.float32,
        )

        self.action_space = spaces.Discrete(4)

        # --------------------------------------------------------------
        # ROS interface
        # --------------------------------------------------------------

        self.ros = ROSInterfaceRunner()

        # --------------------------------------------------------------
        # Episode state
        # --------------------------------------------------------------

        self.current_step = 0
        self.previous_linear_velocity = 0.0

    # ==================================================================
    # Gymnasium API
    # ==================================================================

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """
        Reset the environment and return the first observation.
        """

        super().reset(seed=seed)

        self.current_step = 0
        self.previous_linear_velocity = 0.0

        # Always stop before resetting.
        self.ros.node.stop_robot()

        # Reset the robot's simulator pose.
        #
        # The actual Gazebo pose-reset implementation will be wired
        # through the ROS/Gazebo interface. Keeping this as a separate
        # method prevents simulator-specific logic from spreading
        # through the Gymnasium code.
        self._reset_robot_pose()

        # Wait until fresh sensor data is available.
        self._wait_for_sensor_data()

        scan = self.ros.node.get_latest_scan()

        if scan is None:
            raise RuntimeError("No LiDAR data received during reset.")

        observation = process_lidar_scan(scan)

        self._validate_observation(observation)

        info = {
            "step": self.current_step,
            "collision": False,
        }

        return observation, info

    def step(
        self,
        action: int,
    ) -> tuple[
        np.ndarray,
        float,
        bool,
        bool,
        dict[str, Any],
    ]:
        """
        Apply one discrete action and advance the environment.
        """

        if not self.action_space.contains(action):
            raise ValueError(f"Invalid action: {action}")

        # --------------------------------------------------------------
        # 1. Convert action → velocity
        # --------------------------------------------------------------

        linear_velocity, angular_velocity = self.ACTIONS[int(action)]

        self.ros.node.publish_velocity(
            linear_velocity,
            angular_velocity,
        )

        # --------------------------------------------------------------
        # 2. Allow Gazebo to execute the command
        # --------------------------------------------------------------

        time.sleep(self.CONTROL_DT)

        # --------------------------------------------------------------
        # 3. Read latest sensor data
        # --------------------------------------------------------------

        self._wait_for_sensor_data()

        scan = self.ros.node.get_latest_scan()
        odom = self.ros.node.get_latest_odom()

        if scan is None:
            raise RuntimeError("No LiDAR data received during step().")

        if odom is None:
            raise RuntimeError("No odometry data received during step().")

        # --------------------------------------------------------------
        # 4. Build observation
        # --------------------------------------------------------------

        observation = process_lidar_scan(scan)

        self._validate_observation(observation)

        # --------------------------------------------------------------
        # 5. Collision detection
        # --------------------------------------------------------------

        min_distance = self._minimum_lidar_distance(scan)

        collision = min_distance <= self.COLLISION_DISTANCE

        # --------------------------------------------------------------
        # 6. Reward
        # --------------------------------------------------------------

        forward_velocity = float(
            odom.twist.twist.linear.x
        )

        reward = forward_velocity * self.CONTROL_DT

        if collision:
            reward += self.COLLISION_REWARD

        # --------------------------------------------------------------
        # 7. Episode bookkeeping
        # --------------------------------------------------------------

        self.current_step += 1

        terminated = collision

        truncated = (
            self.current_step >= self.MAX_EPISODE_STEPS
            and not terminated
        )

        # --------------------------------------------------------------
        # 8. Stop immediately after collision
        # --------------------------------------------------------------

        if terminated:
            self.ros.node.stop_robot()

        info = {
            "step": self.current_step,
            "collision": collision,
            "min_lidar_distance": min_distance,
            "forward_velocity": forward_velocity,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # Internal helpers
    # ==================================================================

    def _minimum_lidar_distance(self, scan: Any) -> float:
        """
        Return the minimum valid LiDAR distance.
        """

        ranges = np.asarray(
            scan.ranges,
            dtype=np.float32,
        )

        if ranges.size == 0:
            return LIDAR_MAX_RANGE

        ranges = np.nan_to_num(
            ranges,
            nan=LIDAR_MAX_RANGE,
            posinf=LIDAR_MAX_RANGE,
            neginf=0.0,
        )

        ranges = np.clip(
            ranges,
            0.0,
            LIDAR_MAX_RANGE,
        )

        return float(np.min(ranges))

    def _wait_for_sensor_data(
        self,
        timeout: float = 2.0,
    ) -> None:
        """
        Wait until both LiDAR and odometry data are available.
        """

        start_time = time.monotonic()

        while time.monotonic() - start_time < timeout:
            scan = self.ros.node.get_latest_scan()
            odom = self.ros.node.get_latest_odom()

            if scan is not None and odom is not None:
                return

            time.sleep(0.01)

        raise TimeoutError(
            "Timed out waiting for /scan and /odom."
        )

    def _validate_observation(
        self,
        observation: np.ndarray,
    ) -> None:
        """
        Verify the observation satisfies the environment contract.
        """

        if observation.shape != (NUM_BEAMS,):
            raise ValueError(
                f"Expected observation shape "
                f"{(NUM_BEAMS,)}, got {observation.shape}"
            )

        if observation.dtype != np.float32:
            raise ValueError(
                f"Expected float32 observation, "
                f"got {observation.dtype}"
            )

        if not np.all(np.isfinite(observation)):
            raise ValueError(
                "Observation contains NaN or infinite values."
            )

        if np.any(observation < 0.0) or np.any(observation > 1.0):
            raise ValueError(
                "Observation values must lie in [0, 1]."
            )

    def _reset_robot_pose(self) -> None:
        """
        Reset the Burger to the configured starting pose.

        Simulator-specific pose-reset communication will be implemented
        in the ROS/Gazebo interface so that env.py remains focused on
        Gymnasium semantics.
        """

        self.ros.reset_robot_pose(
            x=self.START_X,
            y=self.START_Y,
            yaw=self.START_YAW,
        )

    # ==================================================================
    # Cleanup
    # ==================================================================

    def close(self) -> None:
        """
        Stop the robot and shut down ROS.
        """

        self.ros.node.stop_robot()
        self.ros.shutdown()
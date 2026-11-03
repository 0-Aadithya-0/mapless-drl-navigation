from __future__ import annotations

import math
from typing import Any

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from .lidar_processing import (
    LIDAR_MAX_RANGE,
    NUM_BEAMS,
    clean_ranges,
    process_lidar_scan,
)
from .ros_interface import ROSInterfaceRunner, stamp_to_ns


class TurtleBotEnv(gym.Env):
    """
    Gymnasium environment for TurtleBot3 obstacle avoidance.

    Week 2 DQN environment:

    Observation:
        24 normalized LiDAR sector minima in [0, 1], sector 0 ahead

    Actions:
        0 = forward
        1 = left
        2 = right
        3 = stop

    Reward:
        forward progress in meters - STEP_COST each step,
        plus COLLISION_REWARD on collision

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

    # Steps are this far apart in simulation time, measured by sensor
    # header stamps. Matches the 5 Hz LiDAR period.
    CONTROL_DT = 0.20

    # The command is published after the previous scan arrives, which
    # is 30-45 ms of sim time after its stamp at 3x real time
    # (rendering + bridge), plus agent compute. Allow this much slack
    # so the next scan on the 5 Hz grid still counts. Too little slack
    # makes steps wait two scan periods.
    #
    # The sim runs freely, so the previous action stays applied until
    # this command reaches Gazebo: each scan-to-scan window starts
    # with that latency under the old action. Only lock-step stepping
    # (pause, then advance exactly CONTROL_DT) removes the overlap.
    STAMP_TOLERANCE = 0.10

    # After a teleport, only sensor data stamped this long after the
    # reset reference is accepted.
    RESET_SETTLE_TIME = 0.10

    COLLISION_DISTANCE = 0.18
    COLLISION_REWARD = -10.0

    # Charged every step. Without it, stopping or spinning in place
    # earns 0 forever and never risks the collision penalty, so idling
    # is a safe local optimum. Idling forever now costs
    # STEP_COST / (1 - gamma) = 1.0 at gamma 0.99: worse than driving,
    # but far less than a collision, so crashing to end an episode
    # never pays.
    STEP_COST = 0.01

    MAX_EPISODE_STEPS = 500

    # Stage 2 center is a safe starting point.
    START_X = 0.0
    START_Y = 0.0
    START_YAW = 0.0
    START_Z = 0.01

    # Wall-clock stall detector only; step timing uses sim stamps.
    SENSOR_TIMEOUT = 10.0

    # ==================================================================
    # Initialization
    # ==================================================================

    def __init__(
        self,
        max_episode_steps: int | None = None,
    ) -> None:
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

        # Odometry and scan stamp from the previous observation, used
        # for the displacement reward and step-duration reporting.
        self._prev_odom = None
        self._prev_scan_stamp_ns = 0

        if max_episode_steps is None:
            self.max_episode_steps = self.MAX_EPISODE_STEPS
        else:
            steps = int(max_episode_steps)

            if steps <= 0 or steps != max_episode_steps:
                raise ValueError(
                    "max_episode_steps must be a positive integer."
                )

            self.max_episode_steps = steps

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

        # Only accept frames rendered after the teleport. Messages
        # already in flight from the old pose carry earlier stamps.
        reset_stamp_ns = self.ros.node.latest_odom_stamp_ns()

        scan, odom = self.ros.node.wait_for_sensor_data(
            min_stamp_ns=reset_stamp_ns
            + self._seconds_to_ns(self.RESET_SETTLE_TIME),
            timeout=self.SENSOR_TIMEOUT,
        )

        self._prev_odom = odom
        self._prev_scan_stamp_ns = stamp_to_ns(scan.header.stamp)

        observation = process_lidar_scan(scan)

        self._validate_observation(observation)

        info = {
            "step": 0,
            "collision": False,
            "min_lidar_distance": self._minimum_lidar_distance(
                scan
            ),
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

        # Sim-time reference for when the command is issued.
        command_stamp_ns = self.ros.node.latest_odom_stamp_ns()

        # Send the action to the robot.
        self.ros.node.publish_velocity(
            linear_x=linear_velocity,
            angular_z=angular_velocity,
        )

        # Wait for the next scan on the CONTROL_DT grid: sensor data
        # stamped at least CONTROL_DT - STAMP_TOLERANCE after the
        # command was issued.
        try:
            scan, odom = self.ros.node.wait_for_sensor_data(
                min_stamp_ns=command_stamp_ns
                + self._seconds_to_ns(
                    self.CONTROL_DT - self.STAMP_TOLERANCE
                ),
                timeout=self.SENSOR_TIMEOUT,
            )
        except BaseException:
            # Never leave the robot driving on a stalled sensor.
            self.ros.node.stop_robot()
            raise

        scan_stamp_ns = stamp_to_ns(scan.header.stamp)

        step_duration = (
            scan_stamp_ns - self._prev_scan_stamp_ns
        ) / 1e9

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

        # Forward displacement over the step, minus the cost of the
        # time spent: a step that makes no progress loses reward.
        progress = self._forward_progress(self._prev_odom, odom)
        reward = progress - self.STEP_COST

        self._prev_odom = odom
        self._prev_scan_stamp_ns = scan_stamp_ns

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
            "step_duration": step_duration,
            "progress": progress,
        }

        return (
            observation,
            float(reward),
            terminated,
            truncated,
            info,
        )

    # ==================================================================
    # Timing and odometry helpers
    # ==================================================================

    @staticmethod
    def _seconds_to_ns(seconds: float) -> int:
        """Convert seconds to integer nanoseconds."""
        return int(round(seconds * 1e9))

    @staticmethod
    def _forward_progress(
        start: Any,
        end: Any,
    ) -> float:
        """
        Return the displacement from start to end odometry, projected
        onto the robot's heading at start, in meters.
        """

        p0 = start.pose.pose.position
        p1 = end.pose.pose.position
        q = start.pose.pose.orientation

        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )

        return (
            (p1.x - p0.x) * math.cos(yaw)
            + (p1.y - p0.y) * math.sin(yaw)
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

        # Same invalid-reading rules as the observation, so a reading
        # inside the minimum range always counts as a collision.
        ranges = clean_ranges(scan)

        if ranges.size == 0:
            return LIDAR_MAX_RANGE

        # Lower bound 0, not the minimum range: a real reading below
        # it is still the closest obstacle.
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
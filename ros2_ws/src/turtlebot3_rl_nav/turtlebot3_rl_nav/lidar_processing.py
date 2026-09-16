from __future__ import annotations
from typing import Final
import numpy as np
from sensor_msgs.msg import LaserScan


NUM_BEAMS: Final[int] = 24
LIDAR_MIN_RANGE: Final[float] = 0.12
LIDAR_MAX_RANGE: Final[float] = 3.5


def process_lidar_scan(
    scan: LaserScan,
    num_beams: int = NUM_BEAMS,
    min_range: float = LIDAR_MIN_RANGE,
    max_range: float = LIDAR_MAX_RANGE,
) -> np.ndarray:
    """
    Convert a ROS LaserScan into a normalized fixed-size LiDAR observation.

    Processing:
        1. Convert ranges to a NumPy array.
        2. Replace NaN and +/-inf with max_range.
        3. Clamp values to the valid LiDAR range.
        4. Downsample the scan to `num_beams`.
        5. Normalize values to [0, 1].

    A normalized value of:
        0.0 -> obstacle at the minimum sensor range
        1.0 -> obstacle at / beyond the maximum sensor range

    Args:
        scan: ROS sensor_msgs/LaserScan message.
        num_beams: Number of output LiDAR beams.
        min_range: Minimum valid LiDAR range in meters.
        max_range: Maximum valid LiDAR range in meters.

    Returns:
        NumPy array with shape (num_beams,) and dtype float32.
    """
    if num_beams <= 0:
        raise ValueError("num_beams must be greater than zero.")

    if min_range >= max_range:
        raise ValueError("min_range must be smaller than max_range.")

    ranges = np.asarray(scan.ranges, dtype=np.float32)

    if ranges.size == 0:
        raise ValueError("LaserScan contains no range measurements.")

    # Invalid measurements mean that no obstacle was detected
    # within the sensor's usable range, so treat them as max range.
    ranges = np.nan_to_num(
        ranges,
        nan=max_range,
        posinf=max_range,
        neginf=min_range,
    )

    # Keep every value inside the physical LiDAR range.
    ranges = np.clip(ranges, min_range, max_range)

    # Downsample while preserving the scan's angular ordering.
    indices = np.linspace(
        0,
        ranges.size - 1,
        num=num_beams,
        dtype=np.int32,
    )
    downsampled = ranges[indices]

    # Normalize [min_range, max_range] → [0, 1].
    normalized = (
        (downsampled - min_range)
        / (max_range - min_range)
    )

    return normalized.astype(np.float32)
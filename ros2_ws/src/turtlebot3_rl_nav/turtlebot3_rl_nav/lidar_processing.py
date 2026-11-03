from __future__ import annotations
from typing import Final
import numpy as np
from sensor_msgs.msg import LaserScan


NUM_BEAMS: Final[int] = 24
LIDAR_MIN_RANGE: Final[float] = 0.12
LIDAR_MAX_RANGE: Final[float] = 3.5


def clean_ranges(
    scan: LaserScan,
    min_range: float = LIDAR_MIN_RANGE,
    max_range: float = LIDAR_MAX_RANGE,
) -> np.ndarray:
    """
    Return the scan's ranges as float32 with non-finite readings replaced.

    Per REP 117, NaN and +inf mean no return (-> max_range), and -inf
    means an object inside the minimum range (-> min_range). Callers
    clip to the bounds they need.
    """
    ranges = np.asarray(scan.ranges, dtype=np.float32)

    return np.nan_to_num(
        ranges,
        nan=max_range,
        posinf=max_range,
        neginf=min_range,
    )


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
        2. Replace NaN/+inf with max_range and -inf with min_range.
        3. Clamp values to the valid LiDAR range.
        4. Split the scan into `num_beams` equal sectors, the first
           centred straight ahead, and keep each sector's minimum.
        5. Normalize values to [0, 1].

    A normalized value of:
        0.0 -> nearest obstacle in the sector at the minimum sensor range
        1.0 -> sector clear out to / beyond the maximum sensor range

    Args:
        scan: ROS sensor_msgs/LaserScan message.
        num_beams: Number of output values (sectors).
        min_range: Minimum valid LiDAR range in meters.
        max_range: Maximum valid LiDAR range in meters.

    Returns:
        NumPy array with shape (num_beams,) and dtype float32.
    """
    if num_beams <= 0:
        raise ValueError("num_beams must be greater than zero.")

    if min_range >= max_range:
        raise ValueError("min_range must be smaller than max_range.")

    ranges = clean_ranges(scan, min_range, max_range)

    if ranges.size < num_beams:
        raise ValueError(
            f"LaserScan has {ranges.size} range measurements; "
            f"at least num_beams ({num_beams}) are needed."
        )

    # Keep every value inside the physical LiDAR range.
    ranges = np.clip(ranges, min_range, max_range)

    # Minimum of each sector, so an obstacle anywhere in a sector shows
    # up; picking one beam per sector misses anything between the picks.
    # Rolling by half a sector puts beam 0 (straight ahead) in the
    # middle of sector 0 instead of on its edge.
    half_sector = ranges.size // (2 * num_beams)
    starts = np.linspace(
        0,
        ranges.size,
        num=num_beams,
        endpoint=False,
    ).astype(np.intp)
    sector_minima = np.minimum.reduceat(
        np.roll(ranges, half_sector),
        starts,
    )

    # Normalize [min_range, max_range] → [0, 1].
    normalized = (
        (sector_minima - min_range)
        / (max_range - min_range)
    )

    return normalized.astype(np.float32)
from types import SimpleNamespace

import numpy as np
import pytest

from turtlebot3_rl_nav.lidar_processing import (
    LIDAR_MAX_RANGE,
    NUM_BEAMS,
    process_lidar_scan,
)


def scan_with(readings: dict[int, float], size: int = 360):
    """A 360-beam scan, clear everywhere except the given beams."""
    ranges = [LIDAR_MAX_RANGE] * size
    for index, value in readings.items():
        ranges[index] = value
    # process_lidar_scan only reads .ranges
    return SimpleNamespace(ranges=ranges)


def test_obstacle_lands_in_its_own_sector():
    # 15 beams per sector, sector k centred on beam 15k: beam 100 is
    # in sector 7 (beams 98-112)
    observation = process_lidar_scan(scan_with({100: 0.5}))

    assert observation.shape == (NUM_BEAMS,)
    assert observation.dtype == np.float32
    assert np.argmin(observation) == 7
    assert np.sum(observation < 1.0) == 1


@pytest.mark.parametrize(
    "beam, sector",
    [
        # sector 0 is beams 353-359 and 0-7, centred straight ahead
        (353, 0), (359, 0), (0, 0), (7, 0),
        # the neighbours just outside it
        (8, 1), (352, 23),
    ],
)
def test_sector_0_is_centred_straight_ahead(beam, sector):
    observation = process_lidar_scan(scan_with({beam: 0.5}))

    assert np.argmin(observation) == sector


def test_non_finite_readings_follow_rep_117():
    # -inf: closer than range_min -> 0.0; NaN/+inf: no return -> 1.0
    observation = process_lidar_scan(
        scan_with({0: float("-inf"), 90: float("nan"), 180: float("inf")})
    )

    assert observation[0] == 0.0
    assert observation[6] == 1.0
    assert observation[12] == 1.0

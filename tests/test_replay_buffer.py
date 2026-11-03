import numpy as np

from rl.agents.replay_buffer import ReplayBuffer


def filled(capacity: int, count: int) -> ReplayBuffer:
    """A buffer with count transitions; transition i is all i's."""
    buffer = ReplayBuffer(capacity, state_dim=3)
    for i in range(count):
        buffer.add([i] * 3, i, float(i), [i + 1] * 3, float(i % 2))
    return buffer


def assert_same(a: ReplayBuffer, b: ReplayBuffer) -> None:
    assert len(a) == len(b)
    assert a.index == b.index
    n = len(a)
    for field in ("states", "actions", "rewards", "next_states", "dones"):
        np.testing.assert_array_equal(
            getattr(a, field)[:n], getattr(b, field)[:n]
        )


def test_full_buffer_round_trip_keeps_its_ring_position():
    # 7 adds into 5 slots wraps the ring: the next write goes to slot 2
    original = filled(capacity=5, count=7)
    restored = ReplayBuffer(5, state_dim=3)
    restored.load_state_dict(original.state_dict())

    assert_same(original, restored)
    assert restored.index == 2


def test_partial_buffer_round_trip_appends_after_the_data():
    original = filled(capacity=5, count=3)
    restored = ReplayBuffer(5, state_dim=3)
    restored.load_state_dict(original.state_dict())

    assert_same(original, restored)

    # the next transition lands after the restored ones, not on them
    restored.add([9] * 3, 9, 9.0, [9] * 3, 0.0)
    assert restored.actions[3] == 9
    np.testing.assert_array_equal(restored.actions[:3], [0, 1, 2])

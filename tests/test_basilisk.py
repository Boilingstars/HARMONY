"""Basilisk backend is optional."""

from harmony.sim.basilisk_world import basilisk_available


def test_basilisk_flag_is_bool():
    assert basilisk_available() in (True, False)

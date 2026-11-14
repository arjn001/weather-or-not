"""
Quick checks on the building blocks. Run from the project folder with:

    python -m pytest

These use small made-up signals, not the dataset, so they finish in about a
second and do not need nuScenes on disk. They do not prove the science is
right. They catch the dull mistakes: a sign flipped, a window off by one,
an attack that silently changes the clean data.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from attacks import drift_attack, offset_attack, replay_attack
from detector import alarm_scores, rolling_median
from residuals import compute_residuals
from scene_io import nearest_resample


def fake_can(seconds=20, hz=50, speed=10.0):
    """A CAN stream at constant speed with a gentle, steadily growing turn."""
    n = seconds * hz
    return {
        "utime": (np.arange(n) * 1e6 / hz).astype(np.int64),
        "speed": np.full(n, speed),
        "yaw_rate": np.linspace(0.0, 0.2, n),
    }


def fake_cam(can, hz=10):
    """A camera that sees exactly what the CAN bus truthfully reports."""
    step = 50 // hz
    return {
        "utime": can["utime"][::step].copy(),
        "first_frame_utime": can["utime"][0],
        "speed": can["speed"][::step].copy(),
        "yaw_rate": can["yaw_rate"][::step].copy(),
    }


# ----------------------------------------------------------------- resampling

def test_nearest_resample_picks_the_closest_reading():
    times = np.array([0, 10, 20, 30])
    values = np.array([0.0, 1.0, 2.0, 3.0])
    picked = nearest_resample(times, values, np.array([1, 9, 14, 16, 29]))
    assert picked.tolist() == [0.0, 1.0, 1.0, 2.0, 3.0]


def test_nearest_resample_never_invents_in_between_values():
    # A jump from 0 to 10. Linear interpolation would produce 5 at the
    # midpoint. Nearest neighbour must return only 0 or 10.
    times = np.array([0, 100])
    values = np.array([0.0, 10.0])
    picked = nearest_resample(times, values, np.arange(0, 101, 7))
    assert set(picked.tolist()) <= {0.0, 10.0}


# -------------------------------------------------------------------- attacks

def test_attacks_leave_the_clean_stream_untouched():
    can = fake_can()
    before = {key: values.copy() for key, values in can.items()}
    offset_attack(can, 5, 10, 0.5)
    drift_attack(can, 5, 10, -0.05)
    replay_attack(can, 10, 15, 0)
    for key in can:
        assert np.array_equal(can[key], before[key])


def test_offset_scales_speed_only_inside_the_window():
    can = fake_can(speed=10.0)
    bad, mask = offset_attack(can, 5, 10, 0.7)
    assert mask.sum() == 5 * 50
    assert np.allclose(bad["speed"][mask], 7.0)
    assert np.allclose(bad["speed"][~mask], 10.0)
    assert np.array_equal(bad["yaw_rate"], can["yaw_rate"])


def test_drift_starts_with_no_jump_and_grows():
    can = fake_can(speed=10.0)
    bad, mask = drift_attack(can, 5, 10, -0.05)
    inside = bad["speed"][mask]
    assert inside[0] == pytest.approx(10.0)          # no jump at the start
    assert inside[-1] == pytest.approx(7.5, abs=0.02)  # 5 s at -5 percent per second
    assert np.all(np.diff(inside) < 0)                # always moving away


def test_replay_copies_earlier_speed_and_yaw():
    can = fake_can()
    can["speed"] = np.linspace(0.0, 20.0, len(can["speed"]))
    bad, mask = replay_attack(can, 10, 15, 2)
    window = np.flatnonzero(mask)
    source = window - 8 * 50  # 10 s minus 2 s, at 50 messages a second
    assert np.array_equal(bad["speed"][window], can["speed"][source])
    assert np.array_equal(bad["yaw_rate"][window], can["yaw_rate"][source])
    assert np.array_equal(bad["speed"][~mask], can["speed"][~mask])


def test_replay_refuses_a_source_that_overlaps_the_window():
    with pytest.raises(ValueError):
        replay_attack(fake_can(), 5, 15, 2)


# ------------------------------------------------------------------ residuals

def test_residuals_are_zero_when_camera_and_can_agree():
    can = fake_can()
    res = compute_residuals(fake_cam(can), can)
    assert np.allclose(res["speed"], 0.0)
    assert np.allclose(res["yaw"], 0.0)


def test_speed_residual_reads_as_a_fraction_with_the_right_sign():
    can = fake_can(speed=10.0)
    cam = fake_cam(can)
    bad, mask = offset_attack(can, 5, 10, 0.7)
    res = compute_residuals(cam, bad)
    inside = (res["t"] >= 5) & (res["t"] < 10)
    # CAN reports 30 percent less than the camera sees.
    assert np.allclose(res["speed"][inside], -0.3)
    assert np.allclose(res["speed"][~inside], 0.0)


def test_speed_residual_stays_bounded_when_the_camera_reads_zero():
    can = fake_can(speed=15.0)
    cam = fake_cam(can)
    cam["speed"][:] = 0.0  # camera has lost the road completely
    res = compute_residuals(cam, can)
    assert np.all(np.abs(res["speed"]) <= 1.0)
    assert np.allclose(res["speed"], 1.0)


# ------------------------------------------------------------------- detector

def test_rolling_median_ignores_a_single_glitch():
    x = np.zeros(50)
    x[30] = 100.0
    smooth = rolling_median(x, 12)
    assert np.isnan(smooth[:11]).all()       # not enough history yet
    assert np.nanmax(smooth) == 0.0          # the glitch is gone


def test_rolling_median_lets_a_sustained_change_through():
    x = np.r_[np.zeros(30), np.ones(30)]
    smooth = rolling_median(x, 12)
    assert smooth[29] == 0.0
    assert smooth[-1] == 1.0


def test_alarm_score_is_high_during_an_attack_and_low_otherwise():
    can = fake_can(speed=10.0)
    cam = fake_cam(can)
    bad, _ = offset_attack(can, 8, 16, 0.7)
    scores = alarm_scores(cam, bad)
    well_inside = (scores["t"] >= 10) & (scores["t"] < 16)
    before = (scores["t"] >= 2) & (scores["t"] < 8)
    assert np.allclose(scores["speed"][well_inside], 0.3)
    assert np.allclose(scores["speed"][before], 0.0)
    assert np.allclose(scores["yaw"][np.isfinite(scores["yaw"])], 0.0)

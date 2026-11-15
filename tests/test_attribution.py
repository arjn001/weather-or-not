"""
Checks for the attribution features, using small made-up signals.

Run from the project folder with:

    python -m pytest tests/test_attribution.py

Nothing here touches the dataset. The fake camera and CAN streams are tiny
and built so the right answer is obvious by hand.
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import attribution_features as af
from attribution_features import (
    HEALTH_KEYS,
    MIN_FINITE,
    WINDOW_FRAMES,
    feature_names,
    health_signals,
    scene_rows,
    window_features,
)


def fake_cam(n=60, hz=10, **overrides):
    """A camera with perfectly steady health signals."""
    cam = {
        "utime": (np.arange(n) * 1e6 / hz).astype(np.int64),
        "first_frame_utime": np.array(0),
        "speed": np.full(n, 10.0),
        "yaw_rate": np.zeros(n),
        "n_tracks": np.full(n, 100.0),
        "n_inliers": np.full(n, 80.0),
        "road_match": np.full(n, 0.9),
        "brightness": np.full(n, 120.0),
        "contrast": np.full(n, 40.0),
        "sharpness": np.full(n, 30.0),
    }
    cam.update(overrides)
    return cam


def fake_res(speed, yaw=None, hz=10):
    n = len(speed)
    return {
        "t": np.arange(n) / hz,
        "speed": np.asarray(speed, dtype=float),
        "yaw": np.zeros(n) if yaw is None else np.asarray(yaw, dtype=float),
    }


def reference_of(cam):
    health = health_signals(cam)
    return {key: float(np.median(health[key])) for key in HEALTH_KEYS}


def test_health_signals_inlier_share_is_a_fraction_per_frame():
    cam = fake_cam(n_tracks=np.array([100.0, 0.0, 50.0]), n_inliers=np.array([80.0, 0.0, 25.0]),
                   utime=np.arange(3))
    health = health_signals(cam)
    assert health["inlier_share"].tolist() == [0.8, 0.0, 0.5]


def test_slope_sign_and_size_on_a_linear_residual():
    # Residual falls by 0.05 every second. Frames are 0.1 s apart.
    n = WINDOW_FRAMES
    speed = -0.05 * (np.arange(n) / 10.0)
    cam = fake_cam(n=n)
    feats = window_features(cam, fake_res(speed), 0, n, reference_of(cam))
    assert feats["res_speed_slope"] < 0
    assert abs(feats["res_speed_slope"] - (-0.05)) < 1e-9


def test_slope_is_positive_for_a_rising_residual():
    n = WINDOW_FRAMES
    speed = 0.02 * (np.arange(n) / 10.0)
    cam = fake_cam(n=n)
    feats = window_features(cam, fake_res(speed), 0, n, reference_of(cam))
    assert abs(feats["res_speed_slope"] - 0.02) < 1e-9


def test_vs_opening_is_one_when_the_signal_never_changes():
    cam = fake_cam(n=60)
    feats = window_features(cam, fake_res(np.full(60, 0.1)), 24, 48, reference_of(cam))
    for key in HEALTH_KEYS:
        assert abs(feats[f"{key}_vs_opening"] - 1.0) < 1e-12


def test_vs_clear_is_half_when_the_window_is_half_the_reference():
    cam = fake_cam(n=48, sharpness=np.full(48, 15.0))
    reference = {key: 1.0 for key in HEALTH_KEYS}
    reference["sharpness"] = 30.0
    feats = window_features(cam, fake_res(np.full(48, 0.1)), 0, 24, reference)
    assert abs(feats["sharpness_vs_clear"] - 0.5) < 1e-12


def test_zero_opening_does_not_divide_by_zero():
    n = 60
    sharp = np.r_[np.zeros(24), np.full(n - 24, 30.0)]
    cam = fake_cam(n=n, sharpness=sharp)
    feats = window_features(cam, fake_res(np.full(n, 0.1)), 24, 48, reference_of(cam))
    assert np.isfinite(feats["sharpness_vs_opening"])


def test_window_with_too_few_finite_values_returns_none():
    speed = np.full(WINDOW_FRAMES, np.nan)
    speed[: MIN_FINITE - 1] = 0.1
    cam = fake_cam(n=WINDOW_FRAMES)
    assert window_features(cam, fake_res(speed), 0, WINDOW_FRAMES, reference_of(cam)) is None


def test_window_with_exactly_enough_finite_values_is_kept():
    speed = np.full(WINDOW_FRAMES, np.nan)
    speed[:MIN_FINITE] = 0.1
    cam = fake_cam(n=WINDOW_FRAMES)
    feats = window_features(cam, fake_res(speed), 0, WINDOW_FRAMES, reference_of(cam))
    assert feats is not None
    assert abs(feats["res_speed_median"] - 0.1) < 1e-12


def test_residual_features_use_signed_and_absolute_values_correctly():
    speed = np.r_[np.full(12, -0.2), np.full(12, -0.4)]
    cam = fake_cam(n=24)
    feats = window_features(cam, fake_res(speed), 0, 24, reference_of(cam))
    assert abs(feats["res_speed_abs_max"] - 0.4) < 1e-12
    assert feats["res_speed_median"] < 0
    assert feats["res_speed_abs_median"] > 0


def test_feature_names_without_health_are_only_residual_names():
    names = feature_names(False)
    assert len(names) > 0
    assert all(n.startswith("res_") for n in names)


def test_feature_names_with_health_adds_both_ratio_groups():
    names = feature_names(True)
    assert len(names) == len(set(names))
    for key in HEALTH_KEYS:
        assert f"{key}_vs_clear" in names
        assert f"{key}_vs_opening" in names
    assert set(feature_names(False)) < set(names)


def test_attack_list_is_exactly_the_agreed_seven():
    assert set(af.ATTACKS) == {
        "offset -20%", "offset -30%", "offset -50%", "offset +20%",
        "offset +30%", "drift -5%/s", "replay from 0s",
    }


def test_attack_windows_only_appear_inside_the_attack_interval(monkeypatch):
    seconds, hz_can, hz_cam = 25, 50, 10
    n_can = seconds * hz_can
    can = {
        "utime": (np.arange(n_can) * 1e6 / hz_can).astype(np.int64),
        "speed": np.full(n_can, 10.0),
        "yaw_rate": np.zeros(n_can),
    }
    cam = fake_cam(n=seconds * hz_cam)
    monkeypatch.setattr(af, "camera_motion_for_scene", lambda name: cam)
    monkeypatch.setattr(af, "load_can_pose", lambda nusc_can, name: can)

    scene = {"name": "scene-fake", "condition": "clear"}
    rows = scene_rows(None, scene, 1.0, reference_of(cam))

    attack_rows = [r for r in rows if r["label"] == "attack"]
    clean_rows = [r for r in rows if r["label"] != "attack"]
    assert len(attack_rows) > 0
    assert len(clean_rows) > 0
    window_seconds = WINDOW_FRAMES / hz_cam
    for r in attack_rows:
        assert r["t_start"] >= af.ATTACK_START - 1e-9
        assert r["t_start"] + window_seconds <= af.ATTACK_END + 0.2
        assert r["attack"] in af.ATTACKS
    assert all(r["attack"] == "none" for r in clean_rows)
    assert all(r["label"] == "nominal" for r in clean_rows)


def test_clean_windows_in_a_rainy_scene_are_labelled_degraded(monkeypatch):
    n_can = 20 * 50
    can = {
        "utime": (np.arange(n_can) * 1e6 / 50).astype(np.int64),
        "speed": np.full(n_can, 10.0),
        "yaw_rate": np.zeros(n_can),
    }
    cam = fake_cam(n=200)
    monkeypatch.setattr(af, "camera_motion_for_scene", lambda name: cam)
    monkeypatch.setattr(af, "load_can_pose", lambda nusc_can, name: can)
    rows = scene_rows(None, {"name": "scene-fake", "condition": "rain"}, 1.0, reference_of(cam))
    clean = [r for r in rows if r["attack"] == "none"]
    assert clean and all(r["label"] == "degraded" for r in clean)


def test_no_attack_windows_when_the_car_is_stopped(monkeypatch):
    n_can = 25 * 50
    can = {
        "utime": (np.arange(n_can) * 1e6 / 50).astype(np.int64),
        "speed": np.zeros(n_can),
        "yaw_rate": np.zeros(n_can),
    }
    cam = fake_cam(n=250, speed=np.zeros(250))
    monkeypatch.setattr(af, "camera_motion_for_scene", lambda name: cam)
    monkeypatch.setattr(af, "load_can_pose", lambda nusc_can, name: can)
    rows = scene_rows(None, {"name": "scene-fake", "condition": "clear"}, 1.0, reference_of(cam))
    assert not any(r["label"] == "attack" for r in rows)


def test_to_matrix_returns_features_and_labels_in_order():
    rows = [{"label": "attack", "a": 1.0, "b": 2.0}, {"label": "nominal", "a": 3.0, "b": 4.0}]
    X, y = af.to_matrix(rows, ["b", "a"])
    assert X.tolist() == [[2.0, 1.0], [4.0, 3.0]]
    assert y.tolist() == ["attack", "nominal"]


# ----------------------------------------------------------- "cannot verify"

def test_gate_never_sees_can_or_the_disagreement():
    # The whole safety argument for the gate rests on this: every input must
    # come from the image alone, so an attacker on the CAN bus cannot move it.
    from verify_gate_eval import GATE_FEATURES

    assert GATE_FEATURES, "the gate needs at least one input"
    for name in GATE_FEATURES:
        assert not name.startswith("res_")
        assert "can" not in name and "speed" not in name and "yaw" not in name


def test_conditions_fold_into_three_reporting_groups():
    from verify_gate_eval import group_of

    assert group_of("clear") == "clear"
    assert group_of("rain") == "rain"
    assert group_of("wet") == "rain"
    assert group_of("night") == "night"
    assert group_of("night+rain") == "night"

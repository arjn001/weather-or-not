"""
Step 4, part 1: turning short windows of driving into features.

The baseline detector looks at one number, the residual (how far CAN is
from the camera). It cannot tell WHY they disagree. Rain and night make the
camera worse, which raises the same alarm as a real attack.

The idea here is to describe a short stretch of driving (a "window") with
two kinds of facts:

  residual features : how big and how steady the disagreement is. An attack
                      tends to give a clean, one-sided offset. A struggling
                      camera tends to give a noisy, jumpy one.
  health features   : how well the camera is doing. How many points it
                      tracked, how sharp the picture is, how bright it is.
                      These are the same for an attack (the camera is fine)
                      and different for rain or night (the camera is not).

Health numbers are always shown as a ratio, so the classifier does not need
to know what a "normal" brightness is. Two ratios are used:

  _vs_clear   : the window compared with a typical clear-weather scene. This
                says "is the camera worse than usual?". It is measured once
                on the training clear scenes only.
  _vs_opening : the window compared with the first second of the SAME scene.
                This says "did the camera get worse during this drive?". An
                attack starts partway through a scene, so the opening
                seconds are almost always clean.

Each window becomes one row with a label:
  nominal  : clear weather, no attack
  degraded : rain, night or wet road, no attack
  attack   : CAN data was falsified (any weather)
"""

import numpy as np

from attacks import drift_attack, offset_attack, replay_attack
from camera_cache import camera_motion_for_scene
from residuals import SPEED_FLOOR, compute_residuals, resample_mask
from scene_io import load_can_pose, nearest_resample

# A window is 24 camera frame pairs, about 2 seconds. Long enough that a
# median ignores single bad frames, short enough that a detector could
# answer within a couple of seconds.
WINDOW_FRAMES = 24
# Slide by half a window so neighbouring windows overlap.
STRIDE = 12
# The first 24 frames of a scene are the "opening" used as a personal
# reference for that scene.
OPENING_FRAMES = 24
# A window needs at least this many frames with a usable speed residual.
MIN_FINITE = 12

HEALTH_KEYS = ("n_tracks", "inlier_share", "road_match", "sharpness", "brightness", "contrast")

# Same attack window as baseline_eval.py.
ATTACK_START, ATTACK_END = 9.0, 16.0

# Used so that dividing by a reference of zero never blows up.
TINY = 1e-9

ATTACKS = {
    "offset -20%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.80),
    "offset -30%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.70),
    "offset -50%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.50),
    "offset +20%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 1.20),
    "offset +30%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 1.30),
    "drift -5%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.05),
    "replay from 0s": lambda can: replay_attack(can, ATTACK_START, ATTACK_END, 0.0),
}

RESIDUAL_NAMES = [
    "res_speed_median",
    "res_speed_abs_median",
    "res_speed_abs_max",
    "res_speed_std",
    "res_speed_slope",
    "res_yaw_abs_median",
    "res_yaw_std",
]


def _nanmedian(x):
    """Median that skips NaN and returns NaN (quietly) when nothing is left."""
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.median(x)) if len(x) else np.nan


def _nanstd(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    return float(np.std(x)) if len(x) else np.nan


def health_signals(cam):
    """
    The six per-frame camera health signals.

    inlier_share is the fraction of tracked points that agreed with the
    final motion estimate. A camera that is guessing has few agreeing
    points even when it tracks many of them.
    """
    n_tracks = np.asarray(cam["n_tracks"], dtype=float)
    n_inliers = np.asarray(cam["n_inliers"], dtype=float)
    return {
        "n_tracks": n_tracks,
        "inlier_share": n_inliers / np.maximum(n_tracks, 1.0),
        "road_match": np.asarray(cam["road_match"], dtype=float),
        "sharpness": np.asarray(cam["sharpness"], dtype=float),
        "brightness": np.asarray(cam["brightness"], dtype=float),
        "contrast": np.asarray(cam["contrast"], dtype=float),
    }


def clear_reference(scene_names):
    """
    A typical value of each health signal in clear weather.

    Takes the median over every frame of the given scenes. Pass ONLY the
    training clear scenes, so the test scenes stay unseen.
    """
    pooled = {key: [] for key in HEALTH_KEYS}
    for name in scene_names:
        health = health_signals(camera_motion_for_scene(name))
        for key in HEALTH_KEYS:
            pooled[key].append(health[key])
    return {key: _nanmedian(np.concatenate(values)) if values else np.nan
            for key, values in pooled.items()}


def _ratio(value, reference):
    """value / reference, with a tiny floor so a zero reference is safe."""
    if not np.isfinite(value) or not np.isfinite(reference):
        return np.nan
    if abs(reference) < TINY:
        reference = TINY
    return float(value / reference)


def window_features(cam, res, start, end, reference, health=None):
    """
    Describe frames [start, end) of one scene as a dict of numbers.

    cam       : camera dict for the scene
    res       : residual dict from compute_residuals (clean or attacked)
    reference : output of clear_reference
    health    : optional precomputed health_signals(cam), to save time when
                many windows are cut from the same scene

    Returns None when fewer than MIN_FINITE frames have a speed residual,
    because statistics from a handful of frames are not worth trusting.
    """
    speed = np.asarray(res["speed"][start:end], dtype=float)
    yaw = np.asarray(res["yaw"][start:end], dtype=float)
    t = np.asarray(res["t"][start:end], dtype=float)
    good = np.isfinite(speed)
    if good.sum() < MIN_FINITE:
        return None

    # Slope of the speed residual over time, in "fraction per second". A
    # drift attack makes this steadily non-zero, a constant offset does not.
    if good.sum() >= 2 and np.ptp(t[good]) > 0:
        slope = float(np.polyfit(t[good], speed[good], 1)[0])
    else:
        slope = np.nan

    feats = {
        "res_speed_median": _nanmedian(speed),
        "res_speed_abs_median": _nanmedian(np.abs(speed)),
        "res_speed_abs_max": float(np.max(np.abs(speed[good]))),
        "res_speed_std": _nanstd(speed),
        "res_speed_slope": slope,
        "res_yaw_abs_median": _nanmedian(np.abs(yaw)),
        "res_yaw_std": _nanstd(yaw),
    }

    if health is None:
        health = health_signals(cam)
    for key in HEALTH_KEYS:
        window_value = _nanmedian(health[key][start:end])
        feats[f"{key}_vs_clear"] = _ratio(window_value, reference[key])
    for key in HEALTH_KEYS:
        window_value = _nanmedian(health[key][start:end])
        opening_value = _nanmedian(health[key][:OPENING_FRAMES])
        feats[f"{key}_vs_opening"] = _ratio(window_value, opening_value)
    return feats


def feature_names(include_health=True):
    """Ordered feature names. With include_health=False, residual ones only."""
    names = list(RESIDUAL_NAMES)
    if include_health:
        names += [f"{key}_vs_clear" for key in HEALTH_KEYS]
        names += [f"{key}_vs_opening" for key in HEALTH_KEYS]
    return names


def _windows(n_frames):
    """Start and end frame of every window in a scene."""
    return [(s, s + WINDOW_FRAMES) for s in range(0, n_frames - WINDOW_FRAMES + 1, STRIDE)]


def scene_rows(nusc_can, scene, speed_scale, reference):
    """
    All labelled windows for one scene: the clean stream, then each attack.

    Clean windows are labelled nominal (clear scene) or degraded (anything
    else). Attack windows are only kept when the attack is active for EVERY
    frame of the window and the car is really moving, since scaling the
    speed of a parked car changes nothing and would be a mislabelled
    example.
    """
    name = scene["name"]
    cam = camera_motion_for_scene(name)
    can = load_can_pose(nusc_can, name)
    health = health_signals(cam)
    n_frames = len(cam["utime"])
    windows = _windows(n_frames)

    def make_row(res, start, end, label, attack_name):
        feats = window_features(cam, res, start, end, reference, health)
        if feats is None:
            return None
        row = {
            "scene": name,
            "condition": scene["condition"],
            "attack": attack_name,
            "label": label,
            "t_start": float(res["t"][start]),
        }
        row.update(feats)
        return row

    rows = []
    clean_res = compute_residuals(cam, can, speed_scale)
    clean_label = "nominal" if scene["condition"] == "clear" else "degraded"
    for start, end in windows:
        row = make_row(clean_res, start, end, clean_label, "none")
        if row is not None:
            rows.append(row)

    true_speed = nearest_resample(can["utime"], can["speed"], cam["utime"])
    for attack_name, attack in ATTACKS.items():
        try:
            bad_can, mask = attack(can)
        except ValueError:
            continue  # e.g. a replay source that does not fit in this scene
        res = compute_residuals(cam, bad_can, speed_scale)
        active = resample_mask(can["utime"], mask, cam["utime"]).astype(bool)
        for start, end in windows:
            if not active[start:end].all():
                continue
            if np.median(true_speed[start:end]) < SPEED_FLOOR:
                continue
            row = make_row(res, start, end, "attack", attack_name)
            if row is not None:
                rows.append(row)
    return rows


def build_rows(nusc_can, scenes, speed_scale, reference):
    """Rows for many scenes, joined into one list."""
    rows = []
    for scene in scenes:
        rows.extend(scene_rows(nusc_can, scene, speed_scale, reference))
    return rows


def to_matrix(rows, names):
    """Rows to (X, y): a float matrix of the chosen features and the labels."""
    X = np.array([[row[n] for n in names] for row in rows], dtype=float).reshape(len(rows), len(names))
    y = np.array([row["label"] for row in rows])
    return X, y

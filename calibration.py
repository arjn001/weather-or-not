"""
Measuring and removing the camera's steady speed over-read.

The road matching method reads speed a few percent high on every scene. The
likely cause is that the camera's true height or tilt differs slightly from
the values in the calibration file, and the speed formula is proportional
to the height.

A steady over-read matters more than it sounds. It gives the detector a
blind direction: an attacker who inflates the reported speed by the same
few percent makes the CAN bus agree with the camera BETTER than the honest
value does.

The fix is one number, the speed scale: typical camera speed divided by
true speed. We measure it once on scenes we trust, the way a car maker
would calibrate a sensor at the factory, and divide it out afterwards.

The important rule: the scenes used to measure the scale must not be the
scenes used to test the detector. Otherwise we would be tuning on the exam.
split_scenes below makes that separation.
"""

import numpy as np

from camera_cache import camera_motion_for_scene
from residuals import SPEED_FLOOR
from scene_io import load_can_pose, nearest_resample

# A scene needs at least this many frames with the car moving before its
# ratio is trusted.
MIN_MOVING_FRAMES = 30


def split_scenes(scenes):
    """
    Split a list of scenes into a calibration half and a test half.

    Scenes are sorted by name and dealt out alternately, like dealing cards
    to two players. That keeps both halves spread across the whole
    recording period instead of one half being all early drives and the
    other all late ones. The split is by whole scene, never by frame:
    frames from one scene are near copies of each other, so sharing a scene
    between the halves would leak answers.
    """
    ordered = sorted(scenes, key=lambda s: s["name"])
    return ordered[0::2], ordered[1::2]


def scene_speed_ratio(nusc_can, scene_name):
    """
    Typical camera speed divided by CAN speed for one clean scene.

    Uses only frames where the car is really moving, and returns NaN if
    there are too few of those.
    """
    cam = camera_motion_for_scene(scene_name)
    can = load_can_pose(nusc_can, scene_name)
    can_speed = nearest_resample(can["utime"], can["speed"], cam["utime"])
    moving = np.isfinite(cam["speed"]) & (can_speed >= SPEED_FLOOR)
    if moving.sum() < MIN_MOVING_FRAMES:
        return np.nan
    return float(np.median(cam["speed"][moving] / can_speed[moving]))


def estimate_speed_scale(nusc_can, scenes):
    """
    The speed scale for a group of trusted, clean scenes.

    Takes each scene's typical ratio, then the median across scenes. Using
    medians twice means neither a few bad frames nor a few odd scenes can
    pull the answer.

    Returns (scale, ratios) where ratios is the per-scene list, NaNs dropped.
    """
    ratios = np.array([scene_speed_ratio(nusc_can, s["name"]) for s in scenes])
    ratios = ratios[np.isfinite(ratios)]
    return float(np.median(ratios)), ratios

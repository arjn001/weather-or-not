"""
Deciding which scenes an experiment may use, and what condition each is in.

A scene is usable when two things hold:
  1. its camera results are in cache/ (so its images were processed once)
  2. its CAN data is complete enough to trust

The second check matters because the nuScenes documentation calls the CAN
data experimental. Some scenes have none, some have gaps. If those slipped
through they would show up later as mysterious NaNs or fake "disagreement"
that has nothing to do with attacks or weather.
"""

import numpy as np

from camera_cache import is_cached
from count_conditions import classify
from scene_io import all_scene_names, load_can_pose, load_scene

# The pose message arrives 50 times a second. We accept a scene if it has
# at least this share of the messages its length calls for.
EXPECTED_HZ = 50
MIN_MESSAGE_SHARE = 0.9
# CAN must start and end within this many seconds of the camera.
MAX_EDGE_GAP = 0.5
# No two consecutive CAN messages may be further apart than this.
MAX_INNER_GAP = 0.5


def condition_of(description):
    """One label per scene: clear, rain, night, night+rain or wet."""
    flags = classify(description)
    if flags["night"] and flags["rain"]:
        return "night+rain"
    if flags["night"]:
        return "night"
    if flags["rain"]:
        return "rain"
    if flags["wet_road"]:
        return "wet"
    return "clear"


def can_problem(nusc_can, scene):
    """
    Return None if the scene's CAN data looks healthy, else a short reason.
    """
    number = int(scene["name"].split("-")[1])
    if number in nusc_can.can_blacklist:
        return "devkit lists this scene as having no CAN data"
    try:
        can = load_can_pose(nusc_can, scene["name"])
    except Exception as error:  # the devkit raises plain Exceptions here
        return f"could not load pose messages ({error})"

    cam_start, cam_end = scene["utimes"][0], scene["utimes"][-1]
    duration = (cam_end - cam_start) / 1e6
    if len(can["utime"]) < MIN_MESSAGE_SHARE * EXPECTED_HZ * duration:
        return f"only {len(can['utime'])} pose messages for {duration:.0f} s"
    if (can["utime"][0] - cam_start) / 1e6 > MAX_EDGE_GAP:
        return "CAN starts late"
    if (cam_end - can["utime"][-1]) / 1e6 > MAX_EDGE_GAP:
        return "CAN ends early"
    if np.diff(can["utime"]).max() / 1e6 > MAX_INNER_GAP:
        return "gap in the middle of the CAN stream"
    if not (np.isfinite(can["speed"]).all() and np.isfinite(can["yaw_rate"]).all()):
        return "CAN contains non-numeric values"
    return None


def usable_scenes(nusc_can, condition=None):
    """
    List the scenes an experiment may use.

    Returns a list of dicts with 'name', 'condition' and 'description'.
    Pass condition="clear" (or "rain", "night", ...) to keep only that kind.
    """
    chosen = []
    for name in all_scene_names():
        if not is_cached(name):
            continue
        scene = load_scene(name)
        label = condition_of(scene["description"])
        if condition is not None and label != condition:
            continue
        if can_problem(nusc_can, scene) is not None:
            continue
        chosen.append({"name": name, "condition": label, "description": scene["description"]})
    return chosen

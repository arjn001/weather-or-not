"""
Remembering camera results so we do not recompute them every time.

Processing one scene takes around ten seconds. Everything after this step
(residuals, attacks, detectors) needs those numbers again and again, so the
first time a scene is processed we save the result to cache/<scene>.npz and
afterwards just load it.

Once a scene is cached, its images are no longer needed by anything in the
project. That is what makes it possible to work through the full dataset on
a laptop: download a batch, process it, delete the images.

Only the CAMERA side is cached. The CAN stream is always loaded fresh and
raw, because attacks are injected into raw CAN before anything else happens
to it.

If you change camera_motion.py or camera_health.py, delete the cache/
folder so results are recomputed with the new code. That needs the images
to be on disk again.
"""

import os

import numpy as np

from camera_motion import estimate_scene_motion
from scene_io import images_on_disk, load_scene

CACHE_DIR = "cache"

# Bump this when the saved contents change, so old files are not mistaken
# for current ones.
CACHE_VERSION = 2


def cache_path(scene_name):
    return os.path.join(CACHE_DIR, f"{scene_name}.npz")


def is_cached(scene_name):
    path = cache_path(scene_name)
    if not os.path.exists(path):
        return False
    with np.load(path) as saved:
        return "cache_version" in saved.files and int(saved["cache_version"]) == CACHE_VERSION


def camera_motion_for_scene(scene_name):
    """
    Return the camera results for a scene, from cache if possible.

    Same dict of arrays that estimate_scene_motion returns, plus
    'first_frame_utime', the timestamp of the scene's first frame, so time
    can be shown as "seconds since the start of the scene".
    """
    if is_cached(scene_name):
        with np.load(cache_path(scene_name)) as saved:
            return {key: saved[key] for key in saved.files}

    scene = load_scene(scene_name)
    if not images_on_disk(scene):
        raise FileNotFoundError(
            f"{scene_name} is not cached and its images are not on disk. "
            "Download the part of the dataset that contains it first."
        )
    cam = estimate_scene_motion(scene["utimes"], scene["paths"], scene["camera"])
    cam["first_frame_utime"] = np.array(scene["utimes"][0])
    cam["cache_version"] = np.array(CACHE_VERSION)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path(scene_name), **cam)
    return cam

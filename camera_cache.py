"""
Remembering camera motion results so we do not recompute them every time.

Estimating motion for one scene takes around ten seconds. Everything after
this step (residuals, attacks, detectors) needs those numbers again and
again, so the first time a scene is processed we save the result to
cache/<scene>.npz and afterwards just load it.

Only the CAMERA side is cached. The CAN stream is always loaded fresh and
raw, because attacks are injected into raw CAN before anything else happens
to it.

If you change camera_motion.py, delete the cache/ folder so results are
recomputed with the new code.
"""

import os

import numpy as np

from camera_motion import estimate_scene_motion
from scene_io import load_camera_frames

CACHE_DIR = "cache"


def camera_motion_for_scene(nusc, scene_name):
    """
    Return the camera motion estimate for a scene, from cache if possible.

    Same dict of arrays that estimate_scene_motion returns, plus
    'first_frame_utime', the timestamp of the scene's first frame, so time
    can be shown as "seconds since the start of the scene".
    """
    path = os.path.join(CACHE_DIR, f"{scene_name}.npz")
    if os.path.exists(path):
        with np.load(path) as saved:
            return {key: saved[key] for key in saved.files}

    utimes, paths, camera = load_camera_frames(nusc, scene_name)
    cam = estimate_scene_motion(utimes, paths, camera)
    cam["first_frame_utime"] = np.array(utimes[0])

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(path, **cam)
    return cam

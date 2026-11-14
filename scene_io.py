"""
Loading one scene into memory.

Two things come out of here:
  1. the front camera frames for a scene, in time order, with timestamps
  2. the CAN "pose" messages for the same scene, as plain numpy arrays

Timestamps everywhere are nuScenes "utime": microseconds since 1970. The
camera and the CAN bus both use this clock, which is what lets us line them
up later.
"""

import os

import numpy as np
from pyquaternion import Quaternion

CAMERA = "CAM_FRONT"


def find_scene(nusc, scene_name):
    """Look a scene up by its name, e.g. 'scene-0061'."""
    for scene in nusc.scene:
        if scene["name"] == scene_name:
            return scene
    raise ValueError(f"No scene called {scene_name} in this dataset version")


def load_camera_frames(nusc, scene_name):
    """
    Return every front camera frame of a scene, oldest first.

    nuScenes splits frames into "samples" (keyframes, about 2 per second,
    the ones humans annotated) and "sweeps" (everything in between). For
    optical flow we want all of them, because the less the image changes
    between two frames, the easier it is to follow points across them.

    The dataset stores frames as a linked list: each record has a 'next'
    token pointing at the frame after it. We start at the scene's first
    keyframe and walk forward until we leave the scene.

    Returns
    -------
    utimes : (N,) int64 array of timestamps in microseconds
    paths  : list of N image file paths
    camera : dict describing the camera, see camera_geometry()
    """
    scene = find_scene(nusc, scene_name)
    first_sample = nusc.get("sample", scene["first_sample_token"])
    token = first_sample["data"][CAMERA]

    utimes, paths = [], []
    calib_token = None
    while token:
        frame = nusc.get("sample_data", token)
        # The chain can run on into the next scene of the same drive, so
        # stop as soon as a frame belongs to a different scene.
        owner = nusc.get("sample", frame["sample_token"])
        if owner["scene_token"] != scene["token"]:
            break
        utimes.append(frame["timestamp"])
        paths.append(os.path.join(nusc.dataroot, frame["filename"]))
        calib_token = frame["calibrated_sensor_token"]
        token = frame["next"]

    camera = camera_geometry(nusc.get("calibrated_sensor", calib_token))
    return np.array(utimes, dtype=np.int64), paths, camera


def camera_geometry(calib):
    """
    Pull out the few facts about the camera that the motion code needs.

    K       : 3x3 intrinsic matrix. Holds the focal length in pixels and
              the pixel where the optical axis hits the image.
    height  : how high the camera sits above the road, in metres.
    forward : the direction the car drives in, as seen from the camera.
    up      : the direction pointing away from the road, as seen from the
              camera.

    The camera is bolted on at a slight tilt, so "forward" is not exactly
    along its optical axis. The calibration stores the camera's orientation
    relative to the car, and we use it to express the car's forward and up
    directions in camera coordinates.
    """
    K = np.array(calib["camera_intrinsic"], dtype=np.float64)
    cam_to_car = Quaternion(calib["rotation"]).rotation_matrix
    forward = cam_to_car.T @ np.array([1.0, 0.0, 0.0])
    up = cam_to_car.T @ np.array([0.0, 0.0, 1.0])

    # Where "straight ahead" lands in the image. Its row is the horizon.
    ahead = K @ forward
    return {
        "K": K,
        "height": float(calib["translation"][2]),
        "forward": forward,
        "up": up,
        "ahead_x": float(ahead[0] / ahead[2]),
        "ahead_y": float(ahead[1] / ahead[2]),
    }


def load_can_pose(nusc_can, scene_name):
    """
    Return the CAN pose stream for a scene as arrays.

    The pose message arrives about 50 times a second. We keep the two
    numbers the project is built on:
      speed    : forward speed in m/s          (vel[0])
      yaw_rate : turn rate in rad/s, left is + (rotation_rate[2])

    Returns a dict with 'utime', 'speed', 'yaw_rate'.
    """
    messages = nusc_can.get_messages(scene_name, "pose")
    return {
        "utime": np.array([m["utime"] for m in messages], dtype=np.int64),
        "speed": np.array([m["vel"][0] for m in messages], dtype=np.float64),
        "yaw_rate": np.array([m["rotation_rate"][2] for m in messages], dtype=np.float64),
    }


def nearest_resample(src_utime, src_values, target_utime):
    """
    Bring a signal onto another timeline by picking the nearest reading.

    For each target time we take the source reading closest to it. We do
    this instead of linear interpolation on purpose: interpolation smooths
    sudden jumps, and a sudden jump is exactly what an injected attack
    looks like. See lesson 4 in section 6 of the brief.
    """
    idx = np.searchsorted(src_utime, target_utime)
    idx = np.clip(idx, 1, len(src_utime) - 1)
    left_is_closer = (target_utime - src_utime[idx - 1]) < (src_utime[idx] - target_utime)
    idx = idx - left_is_closer
    return src_values[idx]

"""
Turning camera and CAN into a disagreement signal.

The "residual" is how far the CAN bus is from what the camera saw. It is
the raw material for every detector in this project. Two channels, kept
separate on purpose (brief section 6, lesson 2):

  speed residual : RELATIVE. (CAN - camera) / (the larger of the two).
                   A value of -0.2 means CAN reports 20 percent less speed
                   than the camera sees. It always lies between -1 and +1.
  yaw residual   : ABSOLUTE, in rad/s. CAN - camera.

Why relative for speed: the camera's speed error is a percentage of the
speed (it reads about 5 percent high at any speed), so a fixed m/s
threshold would be too tight on a fast road and too loose on a slow one.

Why absolute for yaw: turn rate is zero most of the time, and you cannot
take a percentage of zero.

Why divide by the larger of the two: it keeps the number bounded. When the
camera loses the road and reports zero, dividing by the camera alone would
blow up to huge values. This way a total disagreement reads as 1.0 (100
percent) no matter which side is wrong.
"""

import numpy as np

from scene_io import nearest_resample

# Relative error stops making sense when the car is nearly stopped: a
# 0.2 m/s wobble at 0.4 m/s would read as "50 percent". Below this speed we
# divide by the floor instead, so the residual quietly turns into
# (difference in m/s) / SPEED_FLOOR.
SPEED_FLOOR = 2.0  # m/s


def compute_residuals(cam, can, speed_scale=1.0):
    """
    Compare a camera motion estimate against a CAN stream.

    cam : dict from camera_motion_for_scene
    can : dict from load_can_pose, clean or attacked
    speed_scale : how much the camera over-reads speed, e.g. 1.04 for 4
        percent. Camera speed is divided by this before comparing. Get the
        value from calibration.estimate_speed_scale. The default of 1.0
        applies no correction.

    CAN is first brought onto the camera's timeline with nearest-neighbour
    resampling (brief section 6, lesson 4).

    Returns a dict with one entry per camera frame pair:
      t         : seconds since the first frame of the scene
      can_speed, can_yaw : CAN values on the camera timeline
      speed     : relative speed residual (no units)
      yaw       : yaw residual in rad/s
    Entries are NaN where the camera produced no estimate.
    """
    can_speed = nearest_resample(can["utime"], can["speed"], cam["utime"])
    can_yaw = nearest_resample(can["utime"], can["yaw_rate"], cam["utime"])

    cam_speed = cam["speed"] / speed_scale
    ruler = np.maximum(np.maximum(np.abs(cam_speed), np.abs(can_speed)), SPEED_FLOOR)
    return {
        "t": (cam["utime"] - cam["first_frame_utime"]) / 1e6,
        "can_speed": can_speed,
        "can_yaw": can_yaw,
        "speed": (can_speed - cam_speed) / ruler,
        "yaw": can_yaw - cam["yaw_rate"],
    }


def resample_mask(can_utime, mask, cam_utime):
    """Bring a per-CAN-message attack mask onto the camera timeline."""
    return nearest_resample(can_utime, mask, cam_utime)

"""
Injecting falsified values into a CAN stream.

Each function takes a clean CAN stream (the dict from load_can_pose) and
returns two things:
  1. a copy with the attack applied
  2. a True/False array, one entry per CAN message, marking which messages
     were tampered with. This is the ground truth the detector is scored
     against.

The clean stream is never modified. Attack windows are given in seconds
since the first CAN message of the scene.

Threat model: the attacker can rewrite the values the car reports about
itself, but cannot touch the camera image.
"""

import numpy as np


def _seconds(can):
    """Time of each CAN message in seconds since the first one."""
    return (can["utime"] - can["utime"][0]) / 1e6


def _copy(can):
    return {key: values.copy() for key, values in can.items()}


def offset_attack(can, start, end, factor):
    """
    Scale the reported speed by a constant inside the window.

    factor=0.8 means the car claims to be going 20 percent slower than it
    really is. This is the classic "under-report speed" spoof. It switches
    on and off abruptly.
    """
    attacked = _copy(can)
    t = _seconds(can)
    inside = (t >= start) & (t < end)
    attacked["speed"][inside] *= factor
    return attacked, inside


def drift_attack(can, start, end, rate):
    """
    Slowly bend the reported speed away from the truth.

    The scaling starts at exactly 1 (no change) when the window opens and
    moves by `rate` per second. rate=-0.03 means that after 5 seconds the
    car is under-reporting by 15 percent. There is no sudden jump at the
    start, which is what makes this one hard to catch. When the window
    closes the value snaps back to the truth.
    """
    attacked = _copy(can)
    t = _seconds(can)
    inside = (t >= start) & (t < end)
    attacked["speed"][inside] *= 1.0 + rate * (t[inside] - start)
    return attacked, inside


def replay_attack(can, start, end, source_start):
    """
    Overwrite the window with real telemetry recorded earlier in the scene.

    Messages from `source_start` onward are copied over the window. Speed
    and turn rate are both replayed, because a replay copies whole messages.
    Every value is one the car genuinely produced, so nothing looks odd on
    its own. Only the mismatch with what the camera sees gives it away.
    """
    attacked = _copy(can)
    t = _seconds(can)
    inside = np.flatnonzero((t >= start) & (t < end))
    first_source = int(np.searchsorted(t, source_start))
    source = first_source + np.arange(len(inside))
    if len(inside) == 0 or source[-1] >= inside[0]:
        raise ValueError("replay source must lie entirely before the attack window")

    for key in ("speed", "yaw_rate"):
        attacked[key][inside] = can[key][source]

    mask = np.zeros(len(t), dtype=bool)
    mask[inside] = True
    return attacked, mask

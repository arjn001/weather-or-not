"""
The baseline detector: a threshold on the residual.

This is the existing approach from the literature, and the one this project
argues against. It is supposed to work in clear weather. Step 3 shows what
happens to it in rain and at night.

The detector itself is tiny. For every camera frame it produces one number
per channel, an "alarm score": how far CAN is from the camera right now,
smoothed over the last second. An alarm is raised whenever a score is above
a threshold.

We never pick one threshold here. Instead we ask how well the score
separates attacked frames from clean frames across ALL possible thresholds.
That is what AUROC measures: 1.0 means perfect separation, 0.5 means the
score is no better than flipping a coin.
"""

import numpy as np

from residuals import compute_residuals

# Scores are a median over this many frames, roughly one second of video.
# A median over a window ignores single-frame glitches in the camera
# estimate, while a sustained disagreement passes straight through.
SMOOTH_FRAMES = 12


def rolling_median(x, width):
    """
    Median of each value together with the values just before it.

    Looks only backward in time, as a detector running live in a car would
    have to. The first width-1 entries are NaN because there is not enough
    history yet. NaNs inside a window are skipped.
    """
    out = np.full(len(x), np.nan)
    for i in range(width - 1, len(x)):
        window = x[i - width + 1 : i + 1]
        if np.isfinite(window).any():
            out[i] = np.nanmedian(window)
    return out


def alarm_scores(cam, can, speed_scale=1.0, smooth_frames=SMOOTH_FRAMES):
    """
    Turn camera and CAN into per-frame alarm scores.

    Returns a dict with one entry per camera frame pair:
      t     : seconds since the first frame
      speed : size of the smoothed relative speed residual, 0 to 1
      yaw   : size of the smoothed yaw residual, rad/s
    The two channels are kept separate because they are different kinds of
    measurement: yaw needs no depth, speed does.
    """
    res = compute_residuals(cam, can, speed_scale)
    return {
        "t": res["t"],
        "speed": np.abs(rolling_median(res["speed"], smooth_frames)),
        "yaw": np.abs(rolling_median(res["yaw"], smooth_frames)),
    }

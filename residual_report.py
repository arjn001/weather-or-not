"""
Step 2 groundwork: what do residuals look like, clean and under attack?

    python residual_report.py                 # uses scene-0103
    python residual_report.py scene-0655

Does two things:

1. Prints how large the residuals are on every scene when NOTHING is wrong,
   and saves that to out/clean_residuals.csv. This is the noise floor. An
   attack smaller than the noise floor cannot be caught by a threshold.

2. For one scene, applies each of the three attacks to the CAN stream and
   saves out/attacks_<scene>.png, a grid with one column per case (clean,
   offset, drift, replay) and one row each for speed, speed residual and
   yaw residual. The shaded band is where the attack is active.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from attacks import drift_attack, offset_attack, replay_attack
from camera_cache import camera_motion_for_scene
from detector import SMOOTH_FRAMES, rolling_median
from residuals import compute_residuals
from scene_io import DATA_ROOT, load_can_pose, load_scene
from scene_sets import usable_scenes

OUT_DIR = "out"

# The attack window and strengths used for the example figure. These are
# deliberately strong so the effect is easy to see. Step 2 proper will
# sweep over a range of strengths.
ATTACK_START, ATTACK_END = 8.0, 16.0  # seconds into the scene
OFFSET_FACTOR = 0.7  # report 30 percent less speed
DRIFT_RATE = -0.05  # lose 5 percent more per second
REPLAY_SOURCE = 0.0  # replay the scene's opening seconds

def clean_noise_table(nusc_can):
    """Size of the residuals on every usable scene with no attack present."""
    rows = []
    for scene in usable_scenes(nusc_can):
        name = scene["name"]
        cam = camera_motion_for_scene(name)
        res = compute_residuals(cam, load_can_pose(nusc_can, name))
        speed = res["speed"][np.isfinite(res["speed"])]
        yaw = res["yaw"][np.isfinite(res["yaw"])]
        speed_smooth = rolling_median(res["speed"], SMOOTH_FRAMES)
        yaw_smooth = rolling_median(res["yaw"], SMOOTH_FRAMES)
        rows.append({
            "scene": name,
            "condition": scene["condition"],
            # Typical offset: negative means the camera reads faster than CAN.
            "speed_bias": float(np.median(speed)),
            # 95 percent of frames have a residual smaller than this.
            "speed_p95": float(np.percentile(np.abs(speed), 95)),
            # Worst value after one-second smoothing.
            "speed_smooth_max": float(np.nanmax(np.abs(speed_smooth))),
            "yaw_p95": float(np.percentile(np.abs(yaw), 95)),
            "yaw_smooth_max": float(np.nanmax(np.abs(yaw_smooth))),
        })
    return rows


def attack_figure(nusc_can, scene_name):
    cam = camera_motion_for_scene(scene_name)
    clean = load_can_pose(nusc_can, scene_name)
    no_attack = np.zeros(len(clean["utime"]), dtype=bool)

    cases = [
        ("clean", (clean, no_attack)),
        (f"offset (x{OFFSET_FACTOR})",
         offset_attack(clean, ATTACK_START, ATTACK_END, OFFSET_FACTOR)),
        (f"drift ({DRIFT_RATE:+.0%} per second)",
         drift_attack(clean, ATTACK_START, ATTACK_END, DRIFT_RATE)),
        (f"replay (from t={REPLAY_SOURCE:.0f}s)",
         replay_attack(clean, ATTACK_START, ATTACK_END, REPLAY_SOURCE)),
    ]

    fig, axes = plt.subplots(3, 4, figsize=(16, 8.5), sharex=True, sharey="row")
    can_color, cam_color, raw_color = "#1f4e79", "#d1495b", "#bbbbbb"
    t_can = (clean["utime"] - cam["first_frame_utime"]) / 1e6

    for col, (title, (can, mask)) in enumerate(cases):
        res = compute_residuals(cam, can)
        t = res["t"]

        ax = axes[0, col]
        ax.plot(t_can, can["speed"], color=can_color, lw=2, label="CAN (as reported)")
        ax.plot(t, cam["speed"], color=cam_color, lw=1, label="camera")
        ax.set_title(title, fontsize=11)

        ax = axes[1, col]
        ax.plot(t, res["speed"], color=raw_color, lw=1, label="every frame")
        ax.plot(t, rolling_median(res["speed"], SMOOTH_FRAMES), color="black", lw=1.8,
                label="1 s median")
        ax.axhline(0, color="black", lw=0.5)

        ax = axes[2, col]
        ax.plot(t, res["yaw"], color=raw_color, lw=1)
        ax.plot(t, rolling_median(res["yaw"], SMOOTH_FRAMES), color="black", lw=1.8)
        ax.axhline(0, color="black", lw=0.5)
        ax.set_xlabel("time since first frame (s)")

        if mask.any():
            for ax in axes[:, col]:
                ax.axvspan(ATTACK_START, ATTACK_END, color="#e09f3e", alpha=0.18, lw=0)

    axes[0, 0].set_ylabel("speed (m/s)")
    axes[1, 0].set_ylabel("speed residual\n(fraction, CAN vs camera)")
    axes[2, 0].set_ylabel("yaw residual (rad/s)")
    axes[1, 0].set_ylim(-1.05, 1.05)
    axes[0, 0].legend(loc="best", frameon=False, fontsize=9)
    axes[1, 0].legend(loc="upper left", frameon=False, fontsize=9)
    for ax in axes.ravel():
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    description = load_scene(scene_name)["description"][:70]
    fig.suptitle(f"{scene_name}: {description}\nshaded = attack active", fontsize=11)
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f"attacks_{scene_name}.png")
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def main():
    scene_name = sys.argv[1] if len(sys.argv) > 1 else "scene-0103"
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)

    rows = clean_noise_table(nusc_can)
    print("Residuals with no attack present")
    print("scene       condition   speed bias  speed p95  speed 1s max   yaw p95  yaw 1s max")
    for r in rows:
        print(f"{r['scene']}  {r['condition']:10}  {r['speed_bias']:+10.3f}  {r['speed_p95']:9.2f}"
              f"  {r['speed_smooth_max']:12.2f}  {r['yaw_p95']:8.4f}  {r['yaw_smooth_max']:10.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "clean_residuals.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    figure_path = attack_figure(nusc_can, scene_name)
    print(f"\nwrote {csv_path}\nwrote {figure_path}")


if __name__ == "__main__":
    main()

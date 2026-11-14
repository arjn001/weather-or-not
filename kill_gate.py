"""
The kill gate: does the camera agree with the CAN bus on clean data?

Run it on one or more scenes:
    python kill_gate.py scene-0061
    python kill_gate.py scene-0061 scene-0103

or on every scene that has been processed so far:
    python kill_gate.py all

Each scene gets a figure in out/ with three panels on one time axis:
speed, turn rate, and two health signals from the camera side.
A one-line summary per scene is printed. With "all", the summaries are
also saved to out/kill_gate.csv.

Nothing here is fitted or tuned to make the curves match. The camera
estimate is plotted exactly as it comes out of camera_motion.py.
"""

import csv
import os
import sys

import matplotlib

matplotlib.use("Agg")  # write image files, do not open windows
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from camera_cache import camera_motion_for_scene, is_cached
from scene_io import DATA_ROOT, all_scene_names, load_can_pose, load_scene, nearest_resample

OUT_DIR = "out"

# A correlation only means something if the signal actually changes during
# the scene. If the car holds a steady speed or never turns, there is
# nothing to correlate against and we report it as not applicable.
MIN_SPEED_SWING = 1.0  # m/s between the 5th and 95th percentile
MIN_YAW_SWING = 0.05  # rad/s, same idea


def swing(x):
    return np.percentile(x, 95) - np.percentile(x, 5)


def correlation(a, b, ok):
    if ok.sum() < 10:
        return np.nan
    return float(np.corrcoef(a[ok], b[ok])[0, 1])


def run_scene(nusc_can, scene_name):
    scene = load_scene(scene_name)
    cam = camera_motion_for_scene(scene_name)
    can = load_can_pose(nusc_can, scene_name)
    first_frame = cam["first_frame_utime"]

    # Put CAN on the camera's timeline so the two can be compared point by
    # point. Time is shown in seconds from the first camera frame.
    can_speed = nearest_resample(can["utime"], can["speed"], cam["utime"])
    can_yaw = nearest_resample(can["utime"], can["yaw_rate"], cam["utime"])
    t = (cam["utime"] - first_frame) / 1e6
    t_can = (can["utime"] - first_frame) / 1e6

    speed_ok = np.isfinite(cam["speed"])
    yaw_ok = np.isfinite(cam["yaw_rate"])
    moving = speed_ok & (can_speed > 1.0)

    summary = {
        "scene": scene_name,
        "description": scene["description"],
        "frames": len(scene["utimes"]),
        "speed_corr": correlation(cam["speed"], can_speed, speed_ok)
        if swing(can_speed) >= MIN_SPEED_SWING else np.nan,
        # Typical camera speed divided by CAN speed. 1.0 would be perfect.
        "speed_ratio": float(np.median(cam["speed"][moving] / can_speed[moving]))
        if moving.sum() >= 10 else np.nan,
        "speed_mae": float(np.mean(np.abs(cam["speed"][speed_ok] - can_speed[speed_ok]))),
        "yaw_corr": correlation(cam["yaw_rate"], can_yaw, yaw_ok)
        if swing(can_yaw) >= MIN_YAW_SWING else np.nan,
        "yaw_mae": float(np.mean(np.abs(cam["yaw_rate"][yaw_ok] - can_yaw[yaw_ok]))),
        "median_tracks": float(np.median(cam["n_tracks"])),
        "median_road_match": float(np.median(cam["road_match"])),
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True)
    can_color, cam_color = "#1f4e79", "#d1495b"

    axes[0].plot(t_can, can["speed"], color=can_color, lw=2, label="CAN bus")
    axes[0].plot(t, cam["speed"], color=cam_color, lw=1.2, label="camera (road matching)")
    axes[0].set_ylabel("forward speed (m/s)")
    axes[0].set_ylim(bottom=min(-0.5, np.nanmin(cam["speed"]) - 0.5))
    axes[0].legend(loc="best", frameon=False)

    axes[1].plot(t_can, can["yaw_rate"], color=can_color, lw=2, label="CAN bus")
    axes[1].plot(t, cam["yaw_rate"], color=cam_color, lw=1.2, label="camera (tracked points)")
    axes[1].set_ylabel("turn rate (rad/s, left +)")
    axes[1].legend(loc="best", frameon=False)

    # Two signals that say how well the camera side is doing, on separate
    # axes because their scales have nothing to do with each other.
    axes[2].plot(t, cam["n_tracks"], color="#555555", lw=1.2)
    axes[2].set_ylabel("points tracked", color="#555555")
    axes[2].set_ylim(0, 1600)
    axes[2].set_xlabel("time since first frame (s)")
    right = axes[2].twinx()
    right.plot(t, cam["road_match"], color="#e09f3e", lw=1.2)
    right.set_ylabel("road match quality (0 to 1)", color="#b07a1e")
    right.set_ylim(0, 1)
    right.spines["top"].set_visible(False)

    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)

    def fmt(x):
        return "n/a" if np.isnan(x) else f"{x:.2f}"

    fig.suptitle(
        f"{scene_name}: {summary['description'][:70]}\n"
        f"speed corr {fmt(summary['speed_corr'])}, camera/CAN speed ratio "
        f"{fmt(summary['speed_ratio'])}, yaw corr {fmt(summary['yaw_corr'])}",
        fontsize=11,
    )
    fig.tight_layout()
    os.makedirs(OUT_DIR, exist_ok=True)
    fig.savefig(os.path.join(OUT_DIR, f"kill_gate_{scene_name}.png"), dpi=130)
    plt.close(fig)
    return summary


def main():
    if len(sys.argv) < 2:
        sys.exit("usage: python kill_gate.py <scene-name ... | all>")

    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    run_all = sys.argv[1] == "all"
    names = [n for n in all_scene_names() if is_cached(n)] if run_all else sys.argv[1:]

    rows = []
    for name in names:
        row = run_scene(nusc_can, name)
        rows.append(row)
        print(
            f"{row['scene']}  speed corr {row['speed_corr']:5.2f}  "
            f"ratio {row['speed_ratio']:5.2f}  yaw corr {row['yaw_corr']:5.2f}  "
            f"road match {row['median_road_match']:4.2f}  | {row['description'][:45]}",
            flush=True,
        )

    if run_all:
        path = os.path.join(OUT_DIR, "kill_gate.csv")
        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nsummary written to {path}")


if __name__ == "__main__":
    main()

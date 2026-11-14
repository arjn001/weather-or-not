"""
Step 2: how well does the threshold detector work in clear weather?

    python baseline_eval.py

For every clear daytime scene where the car actually moves, and for each
attack type at several strengths, this injects the attack, computes alarm
scores, and measures how well the scores separate attacked frames from
clean ones (AUROC).

Writes out/baseline_auroc.csv and out/baseline_roc.png.

How frames are labelled:
  attacked : the attack is active AND the car is really moving faster than
             2 m/s. Scaling the speed of a stopped car changes nothing, so
             those frames are not counted as attacked.
  clean    : every frame of the same scenes with no attack injected.
Frames in the first second of a scene have no score yet and are left out.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from sklearn.metrics import roc_auc_score, roc_curve

from attacks import drift_attack, offset_attack, replay_attack
from camera_cache import camera_motion_for_scene
from detector import alarm_scores
from residuals import SPEED_FLOOR, resample_mask
from scene_io import DATA_ROOT, load_can_pose, nearest_resample
from scene_sets import usable_scenes

OUT_DIR = "out"

# Every attack runs over the same stretch of each scene.
ATTACK_START, ATTACK_END = 9.0, 16.0

# name -> function that takes a clean CAN stream and returns (attacked, mask)
ATTACKS = {
    "offset -5%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.95),
    "offset -10%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.90),
    "offset -20%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.80),
    "offset -30%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.70),
    "offset -50%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 0.50),
    "drift -1%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.01),
    "drift -2%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.02),
    "drift -5%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.05),
    "replay from 0s": lambda can: replay_attack(can, ATTACK_START, ATTACK_END, 0.0),
}

# The car must be moving for at least this share of the attack window, or
# the scene has nothing to attack and is skipped.
MIN_MOVING_SHARE = 0.5


def collect(nusc_can):
    """
    Gather alarm scores for clean frames and for each attack.

    Returns (clean, attacked, used_scenes) where
      clean    = {"speed": array, "yaw": array}
      attacked = {attack name: {"speed": array, "yaw": array}}
    """
    clean = {"speed": [], "yaw": []}
    attacked = {name: {"speed": [], "yaw": []} for name in ATTACKS}
    used = []

    for scene in usable_scenes(nusc_can, condition="clear"):
        name = scene["name"]
        cam = camera_motion_for_scene(name)
        can = load_can_pose(nusc_can, name)

        true_speed = nearest_resample(can["utime"], can["speed"], cam["utime"])
        t = (cam["utime"] - cam["first_frame_utime"]) / 1e6
        in_window = (t >= ATTACK_START) & (t < ATTACK_END)
        moving = true_speed >= SPEED_FLOOR
        if (moving & in_window).sum() < MIN_MOVING_SHARE * in_window.sum():
            continue
        used.append(name)

        base = alarm_scores(cam, can)
        has_score = np.isfinite(base["speed"]) & np.isfinite(base["yaw"])
        for channel in clean:
            clean[channel].append(base[channel][has_score])

        for attack_name, attack in ATTACKS.items():
            bad_can, mask = attack(can)
            scores = alarm_scores(cam, bad_can)
            active = resample_mask(can["utime"], mask, cam["utime"])
            keep = active & moving & np.isfinite(scores["speed"]) & np.isfinite(scores["yaw"])
            for channel in clean:
                attacked[attack_name][channel].append(scores[channel][keep])

    clean = {c: np.concatenate(v) for c, v in clean.items()}
    attacked = {a: {c: np.concatenate(v) for c, v in d.items()} for a, d in attacked.items()}
    return clean, attacked, used


def auroc(clean_scores, attacked_scores):
    labels = np.r_[np.zeros(len(clean_scores)), np.ones(len(attacked_scores))]
    return float(roc_auc_score(labels, np.r_[clean_scores, attacked_scores]))


def detection_rate(clean_scores, attacked_scores, false_alarm_rate=0.05):
    """Share of attacked frames caught when 5 percent of clean frames alarm."""
    threshold = np.quantile(clean_scores, 1.0 - false_alarm_rate)
    return float(np.mean(attacked_scores > threshold))


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    clean, attacked, used = collect(nusc_can)

    shown = ", ".join(used) if len(used) <= 12 else ", ".join(used[:12]) + ", ..."
    print(f"clear daytime scenes with a moving car: {len(used)}  ({shown})")
    print(f"clean frames: {len(clean['speed'])}\n")
    print("attack           frames  AUROC speed  AUROC yaw  caught at 5% false alarms (speed)")

    rows = []
    for name, scores in attacked.items():
        row = {
            "attack": name,
            "attacked_frames": len(scores["speed"]),
            "auroc_speed": auroc(clean["speed"], scores["speed"]),
            "auroc_yaw": auroc(clean["yaw"], scores["yaw"]),
            "caught_at_5pct_false_alarms": detection_rate(clean["speed"], scores["speed"]),
        }
        rows.append(row)
        print(f"{name:15}  {row['attacked_frames']:6d}  {row['auroc_speed']:11.2f}  "
              f"{row['auroc_yaw']:9.2f}  {row['caught_at_5pct_false_alarms']:8.0%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "baseline_auroc.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ROC curves for the speed channel. Each curve shows the trade between
    # false alarms (x) and attacks caught (y) as the threshold slides.
    fig, ax = plt.subplots(figsize=(6.5, 6))
    colors = plt.cm.viridis(np.linspace(0.05, 0.9, len(attacked)))
    for (name, scores), color in zip(attacked.items(), colors):
        labels = np.r_[np.zeros(len(clean["speed"])), np.ones(len(scores["speed"]))]
        fpr, tpr, _ = roc_curve(labels, np.r_[clean["speed"], scores["speed"]])
        style = "--" if name.startswith("drift") else ":" if name.startswith("replay") else "-"
        ax.plot(fpr, tpr, style, color=color, lw=1.8,
                label=f"{name} ({auroc(clean['speed'], scores['speed']):.2f})")
    ax.plot([0, 1], [0, 1], color="#aaaaaa", lw=1)
    ax.set_xlabel("false alarm rate (clean frames flagged)")
    ax.set_ylabel("detection rate (attacked frames flagged)")
    ax.set_title(f"Baseline threshold detector, speed channel\n"
                 f"{len(used)} clear daytime scenes", fontsize=11)
    ax.legend(title="attack (AUROC)", frameon=False, fontsize=9, loc="lower right")
    ax.grid(alpha=0.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "baseline_roc.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()

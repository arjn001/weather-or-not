"""
Step 2: how well does the threshold detector work in clear weather?

    python baseline_eval.py

Clear daytime scenes are split into two halves by whole scene:

  calibration half : used to measure the camera's speed over-read and to
                     pick the alarm threshold. No attacks are needed here,
                     only clean driving.
  test half        : never looked at until the end. Attacks are injected
                     into these scenes and the detector is scored on them.

For each attack type and strength this reports, on the test half only:
  AUROC raw        : separation with no over-read correction
  AUROC corrected  : separation after dividing out the over-read
  caught           : share of attacked frames above the threshold
The threshold is the level that 5 percent of clean CALIBRATION frames
exceed. The share of clean TEST frames above it is printed too. If the
detector transfers well between scenes, that number is also close to 5.

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
from calibration import estimate_speed_scale, split_scenes
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
    # Over-reporting. The camera reads a few percent fast on its own, so an
    # attack in this direction first has to cancel that before it shows.
    "offset +5%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 1.05),
    "offset +10%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 1.10),
    "offset +20%": lambda can: offset_attack(can, ATTACK_START, ATTACK_END, 1.20),
    "drift -1%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.01),
    "drift -2%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.02),
    "drift -5%/s": lambda can: drift_attack(can, ATTACK_START, ATTACK_END, -0.05),
    "replay from 0s": lambda can: replay_attack(can, ATTACK_START, ATTACK_END, 0.0),
}

# The car must be moving for at least this share of the attack window, or
# the scene has nothing to attack and is skipped.
MIN_MOVING_SHARE = 0.5


# Share of clean calibration frames allowed to raise an alarm.
FALSE_ALARM_TARGET = 0.05


def collect(nusc_can, scenes, speed_scale):
    """
    Gather alarm scores for clean frames and for each attack.

    Returns (clean, attacked, used_scenes) where
      clean    = {"speed": array, "yaw": array}
      attacked = {attack name: {"speed": array, "yaw": array}}
    Scenes where the car is mostly stopped during the attack window are
    skipped and do not appear in used_scenes.
    """
    clean = {"speed": [], "yaw": []}
    attacked = {name: {"speed": [], "yaw": []} for name in ATTACKS}
    used = []

    for scene in scenes:
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

        base = alarm_scores(cam, can, speed_scale)
        has_score = np.isfinite(base["speed"]) & np.isfinite(base["yaw"])
        for channel in clean:
            clean[channel].append(base[channel][has_score])

        for attack_name, attack in ATTACKS.items():
            bad_can, mask = attack(can)
            scores = alarm_scores(cam, bad_can, speed_scale)
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


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    calibration_scenes, test_scenes = split_scenes(usable_scenes(nusc_can, condition="clear"))

    # Everything learned from data is learned on the calibration half.
    speed_scale, ratios = estimate_speed_scale(nusc_can, calibration_scenes)
    calib_clean, _, calib_used = collect(nusc_can, calibration_scenes, speed_scale)
    threshold = float(np.quantile(calib_clean["speed"], 1.0 - FALSE_ALARM_TARGET))

    # The test half is scored twice: without and with the correction.
    raw_clean, raw_attacked, used = collect(nusc_can, test_scenes, 1.0)
    clean, attacked, _ = collect(nusc_can, test_scenes, speed_scale)
    test_false_alarms = float(np.mean(clean["speed"] > threshold))

    print(f"calibration half: {len(calibration_scenes)} clear scenes, {len(calib_used)} with a moving car")
    print(f"test half:        {len(test_scenes)} clear scenes, {len(used)} with a moving car")
    print(f"camera speed over-read measured on calibration half: {speed_scale:.3f} "
          f"(from {len(ratios)} scenes)")
    print(f"alarm threshold on the speed score: {threshold:.3f}")
    print(f"clean frames above it: {FALSE_ALARM_TARGET:.0%} on calibration (by construction), "
          f"{test_false_alarms:.1%} on test\n")
    print("attack           frames  AUROC raw  AUROC corrected  AUROC yaw  caught")

    rows = []
    for name, scores in attacked.items():
        row = {
            "attack": name,
            "attacked_frames": len(scores["speed"]),
            "auroc_speed_raw": auroc(raw_clean["speed"], raw_attacked[name]["speed"]),
            "auroc_speed_corrected": auroc(clean["speed"], scores["speed"]),
            "auroc_yaw": auroc(clean["yaw"], scores["yaw"]),
            "caught_at_threshold": float(np.mean(scores["speed"] > threshold)),
        }
        rows.append(row)
        print(f"{name:15}  {row['attacked_frames']:6d}  {row['auroc_speed_raw']:9.2f}  "
              f"{row['auroc_speed_corrected']:15.2f}  {row['auroc_yaw']:9.2f}  "
              f"{row['caught_at_threshold']:6.0%}")
    rows.append({"attack": "(none, clean test frames)", "attacked_frames": len(clean["speed"]),
                 "auroc_speed_raw": "", "auroc_speed_corrected": "", "auroc_yaw": "",
                 "caught_at_threshold": test_false_alarms})

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
    ax.set_title(f"Baseline threshold detector, speed channel, over-read corrected\n"
                 f"{len(used)} held-out clear daytime scenes", fontsize=11)
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

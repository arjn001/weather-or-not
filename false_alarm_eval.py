"""
Step 3: what happens to the baseline detector in rain and at night?

    python false_alarm_eval.py

The detector is set up exactly as in Step 2, using clear daytime scenes
only: the camera over-read and the alarm thresholds come from the clear
calibration half. Nothing is retuned here. That is the point. A detector
deployed in a car does not get to pick a new threshold when it starts to
rain.

Then, for each condition (held-out clear, rain, night, night+rain):

  1. NO attack is injected. Every alarm is therefore a false alarm. We
     report how often the detector fires, per frame and per scene.
  2. Attacks ARE injected, and we check whether the alarm score can still
     tell attacked frames from clean frames of the same condition.

Writes out/false_alarms.csv and out/false_alarms.png.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

from baseline_eval import FALSE_ALARM_TARGET, auroc, collect
from calibration import estimate_speed_scale, split_scenes
from camera_cache import camera_motion_for_scene
from detector import alarm_scores
from scene_io import DATA_ROOT, load_can_pose
from scene_sets import usable_scenes

OUT_DIR = "out"
CONDITIONS = ("rain", "night", "night+rain")

# A scene counts as "falsely alarmed" if the detector fires on more than
# this share of its frames. One stray frame should not condemn a scene.
SCENE_ALARM_SHARE = 0.10

# Attacks whose detectability is compared across conditions.
REFERENCE_ATTACKS = ("offset -20%", "offset -30%", "offset +20%", "drift -5%/s", "replay from 0s")


def clean_scores_per_scene(nusc_can, scenes, speed_scale):
    """Alarm scores for each scene with no attack. Returns a list of dicts."""
    out = []
    for scene in scenes:
        cam = camera_motion_for_scene(scene["name"])
        scores = alarm_scores(cam, load_can_pose(nusc_can, scene["name"]), speed_scale)
        ok = np.isfinite(scores["speed"]) & np.isfinite(scores["yaw"])
        out.append({"name": scene["name"], "speed": scores["speed"][ok], "yaw": scores["yaw"][ok]})
    return out


def false_alarm_summary(per_scene, speed_threshold, yaw_threshold):
    """How often the detector fires on scenes that contain no attack."""
    speed = np.concatenate([s["speed"] for s in per_scene])
    yaw = np.concatenate([s["yaw"] for s in per_scene])
    fires = [(s["speed"] > speed_threshold) | (s["yaw"] > yaw_threshold) for s in per_scene]
    return {
        "scenes": len(per_scene),
        "clean_frames": len(speed),
        "frames_alarm_speed": float(np.mean(speed > speed_threshold)),
        "frames_alarm_yaw": float(np.mean(yaw > yaw_threshold)),
        "frames_alarm_either": float(np.mean(np.concatenate(fires))),
        "scenes_alarmed": float(np.mean([f.mean() > SCENE_ALARM_SHARE for f in fires])),
        "median_speed_score": float(np.median(speed)),
    }


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    calibration_scenes, clear_test = split_scenes(usable_scenes(nusc_can, condition="clear"))

    # Set the detector up on clear calibration scenes, exactly as in Step 2.
    speed_scale, _ = estimate_speed_scale(nusc_can, calibration_scenes)
    calib = clean_scores_per_scene(nusc_can, calibration_scenes, speed_scale)
    speed_threshold = float(np.quantile(np.concatenate([s["speed"] for s in calib]),
                                        1.0 - FALSE_ALARM_TARGET))
    yaw_threshold = float(np.quantile(np.concatenate([s["yaw"] for s in calib]),
                                      1.0 - FALSE_ALARM_TARGET))
    print(f"set up on {len(calibration_scenes)} clear calibration scenes")
    print(f"speed over-read {speed_scale:.3f}, speed threshold {speed_threshold:.3f}, "
          f"yaw threshold {yaw_threshold:.4f} rad/s\n")

    groups = {"clear (held out)": clear_test}
    for condition in CONDITIONS:
        scenes = usable_scenes(nusc_can, condition=condition)
        if scenes:
            groups[condition] = scenes

    rows, clean_by_group = [], {}
    for label, scenes in groups.items():
        per_scene = clean_scores_per_scene(nusc_can, scenes, speed_scale)
        clean_by_group[label] = per_scene
        row = {"condition": label, **false_alarm_summary(per_scene, speed_threshold, yaw_threshold)}

        # Can the score still separate attacked from clean within this condition?
        clean, attacked, used = collect(nusc_can, scenes, speed_scale)
        row["scenes_with_moving_car"] = len(used)
        for attack in REFERENCE_ATTACKS:
            channel = "yaw" if attack.startswith("replay") else "speed"
            row[f"auroc {attack}"] = auroc(clean[channel], attacked[attack][channel])
            row[f"caught {attack}"] = float(np.mean(attacked[attack][channel] > (
                yaw_threshold if channel == "yaw" else speed_threshold)))
        rows.append(row)

    print("FALSE ALARMS with no attack present")
    print("condition          scenes  frames  speed alarm  yaw alarm  either  scenes alarmed")
    for r in rows:
        print(f"{r['condition']:17}  {r['scenes']:6d}  {r['clean_frames']:6d}  "
              f"{r['frames_alarm_speed']:10.1%}  {r['frames_alarm_yaw']:9.1%}  "
              f"{r['frames_alarm_either']:6.1%}  {r['scenes_alarmed']:13.0%}")

    print("\nATTACK SEPARATION within each condition (AUROC, then share caught at the clear threshold)")
    print("condition          " + "  ".join(f"{a:>14}" for a in REFERENCE_ATTACKS))
    for r in rows:
        print(f"{r['condition']:17}  " + "  ".join(
            f"{r[f'auroc {a}']:6.2f} {r[f'caught {a}']:6.0%} " for a in REFERENCE_ATTACKS))

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "false_alarms.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Figure. Left: how often the detector fires with nothing wrong.
    # Right: the spread of each scene's typical-worst speed score.
    labels = [r["condition"] for r in rows]
    colors = ["#1f4e79"] + ["#d1495b"] * (len(rows) - 1)
    fig, (left, right) = plt.subplots(1, 2, figsize=(12, 4.8))

    x = np.arange(len(rows))
    left.bar(x - 0.2, [r["frames_alarm_either"] * 100 for r in rows], 0.4, color=colors,
             label="frames")
    left.bar(x + 0.2, [r["scenes_alarmed"] * 100 for r in rows], 0.4, color=colors, alpha=0.45,
             label=f"scenes (alarm on more than {SCENE_ALARM_SHARE:.0%} of frames)")
    left.axhline(FALSE_ALARM_TARGET * 100, color="black", lw=0.8, ls="--")
    left.text(len(rows) - 0.55, FALSE_ALARM_TARGET * 100 + 1.5, "design target", ha="right", fontsize=9)
    left.set_xticks(x)
    left.set_xticklabels([f"{l}\n({r['scenes']} scenes)" for l, r in zip(labels, rows)])
    left.set_ylabel("false alarms, no attack present (%)")
    left.set_title("How often the detector fires with nothing wrong", fontsize=11)
    left.legend(frameon=False, fontsize=9)

    per_scene_p95 = [[np.percentile(s["speed"], 95) for s in clean_by_group[l]] for l in labels]
    boxes = right.boxplot(per_scene_p95, patch_artist=True, widths=0.55, showfliers=True,
                          flierprops=dict(markersize=3))
    for patch, color in zip(boxes["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.55)
    for median in boxes["medians"]:
        median.set_color("black")
    right.axhline(speed_threshold, color="black", lw=0.8, ls="--")
    right.text(len(rows) + 0.45, speed_threshold + 0.015, "alarm threshold", ha="right", fontsize=9)
    right.set_xticks(np.arange(1, len(rows) + 1))
    right.set_xticklabels(labels)
    right.set_ylabel("speed alarm score\n(each scene's 95th percentile)")
    right.set_ylim(0, 1.05)
    right.set_title("Camera vs CAN disagreement per scene, no attack", fontsize=11)

    for ax in (left, right):
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("Baseline threshold detector, set up on clear daytime scenes only", fontsize=12)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "false_alarms.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()

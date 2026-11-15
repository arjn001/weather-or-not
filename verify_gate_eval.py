"""
Step 4, second design: a detector that knows when it cannot check.

    python verify_gate_eval.py

The three-way classifier in attribution_eval.py has a weakness. It decides
"this is just bad weather" partly from how the disagreement looks, so an
attacker who strikes in the dark gets the benefit of the doubt.

This design splits the job in two.

  1. THE GATE looks only at the image (sharpness, brightness, how many
     points could be tracked, and so on) and answers one question: can the
     camera measure the car's motion well enough right now to check the CAN
     bus? If not, the answer for this window is "cannot verify".

  2. THE DETECTOR only runs on windows the gate lets through. It is the
     plain threshold detector from Step 2, unchanged.

Why this is safer. The gate never sees the CAN bus or the disagreement. An
attacker who can only rewrite CAN messages cannot change the image, so they
cannot talk the system into "cannot verify", and they cannot hide behind
bad conditions inside the windows that are checked.

How the gate learns. On clean driving we know the truth: if the camera and
the honest CAN bus agree in a window, the camera was working there. So the
gate is trained on clean windows only, to predict "the camera agrees with
honest CAN" from image measurements alone. No attacks are needed to train
it.

What to look at in the output:
  coverage : share of windows the gate lets through, per condition. The
             price of honesty. Low coverage at night means the system
             admits it is mostly blind there.
  false alarms and attacks caught, inside the windows that were checked.

Everything is repeated over several random train/test splits by whole
scene. Writes out/verify_gate.csv and out/verify_gate.png.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from sklearn.ensemble import HistGradientBoostingClassifier

from attribution_features import HEALTH_KEYS, build_rows, clear_reference, to_matrix
from attribution_robustness import random_split
from calibration import estimate_speed_scale
from scene_io import DATA_ROOT
from scene_sets import usable_scenes

OUT_DIR = "out"
N_REPEATS = 8
BASELINE_QUANTILE = 0.95

# The gate may only use measurements that come from the image alone.
GATE_FEATURES = [f"{key}_vs_clear" for key in HEALTH_KEYS]

# The gate lets a window through when it is at least this sure the camera
# is working. Raising it checks fewer windows but checks them more safely.
GATE_CONFIDENCE = 0.5

GROUPS = ("clear", "rain", "night")


def group_of(condition):
    """Fold night+rain into night, and wet into rain, for reporting."""
    if condition.startswith("night"):
        return "night"
    return "rain" if condition in ("rain", "wet") else "clear"


def share(mask, values):
    return float(np.mean(values[mask])) if mask.any() else np.nan


def one_repeat(nusc_can, scenes, seed, confidence=GATE_CONFIDENCE):
    rng = np.random.default_rng(seed)
    train_scenes, test_scenes = random_split(scenes, rng)
    train_clear = [s for s in train_scenes if s["condition"] == "clear"]
    speed_scale, _ = estimate_speed_scale(nusc_can, train_clear)
    reference = clear_reference([s["name"] for s in train_clear])
    train = build_rows(nusc_can, train_scenes, speed_scale, reference)
    test = build_rows(nusc_can, test_scenes, speed_scale, reference)

    # Detector thresholds, from clear clean training windows, as in Step 2.
    nominal_train = [r for r in train if r["label"] == "nominal"]
    thr_speed = np.quantile([r["res_speed_abs_median"] for r in nominal_train], BASELINE_QUANTILE)
    thr_yaw = np.quantile([r["res_yaw_abs_median"] for r in nominal_train], BASELINE_QUANTILE)

    def alarm(rows):
        return np.array([(r["res_speed_abs_median"] > thr_speed) or (r["res_yaw_abs_median"] > thr_yaw)
                         for r in rows])

    # Train the gate on CLEAN training windows from every condition. The
    # target is whether the camera agreed with the honest CAN bus there.
    clean_train = [r for r in train if r["label"] != "attack"]
    camera_worked = ~alarm(clean_train)
    X_gate, _ = to_matrix(clean_train, GATE_FEATURES)
    gate = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    gate.fit(X_gate, camera_worked)

    X_test, y_test = to_matrix(test, GATE_FEATURES)
    checked = gate.predict_proba(X_test)[:, list(gate.classes_).index(True)] >= confidence
    alarmed = alarm(test)
    groups = np.array([group_of(r["condition"]) for r in test])
    is_attack = y_test == "attack"

    result = {"seed": seed}
    for g in GROUPS + ("all",):
        here = np.ones(len(test), bool) if g == "all" else groups == g
        clean, attacked = here & ~is_attack, here & is_attack
        result[f"{g}: coverage of clean windows"] = share(clean, checked)
        result[f"{g}: coverage of attack windows"] = share(attacked, checked)
        result[f"{g}: false alarms, no gate"] = share(clean, alarmed)
        result[f"{g}: false alarms, in checked windows"] = share(clean & checked, alarmed)
        result[f"{g}: false alarms, with gate, of all windows"] = share(clean, alarmed & checked)
        result[f"{g}: attacks caught, no gate"] = share(attacked, alarmed)
        result[f"{g}: attacks caught, in checked windows"] = share(attacked & checked, alarmed)
    return result


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    scenes = usable_scenes(nusc_can)
    results = [one_repeat(nusc_can, scenes, seed) for seed in range(N_REPEATS)]
    keys = [k for k in results[0] if k != "seed"]
    mean = {k: float(np.nanmean([r[k] for r in results])) for k in keys}
    spread = {k: float(np.nanstd([r[k] for r in results])) for k in keys}

    print(f"{N_REPEATS} random train/test splits of {len(scenes)} scenes, "
          f"gate confidence {GATE_CONFIDENCE}\n")
    measures = [k.split(": ", 1)[1] for k in keys if k.startswith("clear: ")]
    print(f"{'':44}" + "".join(f"{g:>16}" for g in GROUPS + ("all",)))
    for m in measures:
        cells = "".join(f"{mean[f'{g}: {m}']:9.1%} +/-{spread[f'{g}: {m}']:4.1%}" for g in GROUPS + ("all",))
        print(f"{m:44}{cells}")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "verify_gate.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["measure", "mean", "spread"])
        for k in keys:
            writer.writerow([k, mean[k], spread[k]])

    # Figure: per condition, false alarms without the gate, false alarms
    # inside checked windows, and how much the gate lets through.
    x = np.arange(len(GROUPS))
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.6))
    bars = [("false alarms, no gate", "#d1495b", "no gate"),
            ("false alarms, in checked windows", "#1f4e79", "inside checked windows")]
    for i, (m, color, label) in enumerate(bars):
        left.bar(x + (i - 0.5) * 0.38, [mean[f"{g}: {m}"] * 100 for g in GROUPS], 0.38,
                 yerr=[spread[f"{g}: {m}"] * 100 for g in GROUPS], color=color, label=label, capsize=3)
    left.set_xticks(x)
    left.set_xticklabels(GROUPS)
    left.set_ylabel("clean windows that raise an alarm (%)")
    left.set_title("False alarms with nothing wrong", fontsize=11)
    left.legend(frameon=False, fontsize=9)

    bars = [("coverage of clean windows", "#888888", "windows the gate lets through"),
            ("attacks caught, in checked windows", "#1f4e79", "attacks caught inside them")]
    for i, (m, color, label) in enumerate(bars):
        right.bar(x + (i - 0.5) * 0.38, [mean[f"{g}: {m}"] * 100 for g in GROUPS], 0.38,
                  yerr=[spread[f"{g}: {m}"] * 100 for g in GROUPS], color=color, label=label, capsize=3)
    right.set_xticks(x)
    right.set_xticklabels(GROUPS)
    right.set_ylabel("%")
    right.set_ylim(0, 105)
    right.set_title("What the system can check, and how well", fontsize=11)
    right.legend(frameon=False, fontsize=9, loc="lower left")

    for ax in (left, right):
        ax.grid(alpha=0.25, axis="y")
        ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle('Threshold detector with an image-only "can I check?" gate '
                 f"(mean of {N_REPEATS} splits, bars show spread)", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "verify_gate.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()

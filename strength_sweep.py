"""
How strong does an attack have to be before the baseline notices it?

    python strength_sweep.py

Two sweeps on held-out clear daytime scenes, using the Step 2 threshold
detector exactly as set up in baseline_eval.py (over-read and threshold
from the calibration half, about 5 percent false alarms):

  1. Offset attacks from 2 to 50 percent, under-reporting and
     over-reporting.
  2. Drift attacks from 0.25 to 10 percent per second, running for the
     7 second attack window.

For each strength it reports the share of attacked frames that raise an
alarm. The drift sweep answers a practical question: how slowly must an
attacker ramp to stay under the detector?

Writes out/strength_sweep.csv and out/strength_sweep.png.
"""

import csv
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus

import baseline_eval
from attacks import drift_attack, offset_attack
from baseline_eval import ATTACK_END, ATTACK_START, FALSE_ALARM_TARGET, collect
from calibration import estimate_speed_scale, split_scenes
from scene_io import DATA_ROOT
from scene_sets import usable_scenes

OUT_DIR = "out"
OFFSETS = (0.02, 0.05, 0.075, 0.10, 0.125, 0.15, 0.20, 0.30, 0.50)
DRIFT_RATES = (0.0025, 0.005, 0.01, 0.015, 0.02, 0.03, 0.05, 0.075, 0.10)
WINDOW_SECONDS = ATTACK_END - ATTACK_START


def main():
    attacks = {}
    for size in OFFSETS:
        attacks[("under", size)] = lambda can, f=1 - size: offset_attack(can, ATTACK_START, ATTACK_END, f)
        attacks[("over", size)] = lambda can, f=1 + size: offset_attack(can, ATTACK_START, ATTACK_END, f)
    for rate in DRIFT_RATES:
        attacks[("drift", rate)] = lambda can, r=-rate: drift_attack(can, ATTACK_START, ATTACK_END, r)
    # collect() reads its attack list from baseline_eval, so hand it ours.
    baseline_eval.ATTACKS = attacks

    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    calibration_scenes, test_scenes = split_scenes(usable_scenes(nusc_can, condition="clear"))
    speed_scale, _ = estimate_speed_scale(nusc_can, calibration_scenes)
    calib_clean, _, _ = collect(nusc_can, calibration_scenes, speed_scale)
    threshold = float(np.quantile(calib_clean["speed"], 1.0 - FALSE_ALARM_TARGET))
    clean, attacked, used = collect(nusc_can, test_scenes, speed_scale)

    false_alarms = float(np.mean(clean["speed"] > threshold))
    print(f"{len(used)} held-out clear scenes, threshold {threshold:.3f}, "
          f"false alarms on clean frames {false_alarms:.1%}\n")

    rows = []
    print("offset size   under-report caught   over-report caught")
    for size in OFFSETS:
        under = float(np.mean(attacked[("under", size)]["speed"] > threshold))
        over = float(np.mean(attacked[("over", size)]["speed"] > threshold))
        rows.append({"kind": "offset", "strength": size, "under_caught": under, "over_caught": over})
        print(f"{size:10.1%}   {under:19.0%}   {over:18.0%}")

    print(f"\ndrift per second   error after {WINDOW_SECONDS:.0f} s   frames caught")
    for rate in DRIFT_RATES:
        caught = float(np.mean(attacked[("drift", rate)]["speed"] > threshold))
        rows.append({"kind": "drift", "strength": rate, "under_caught": caught, "over_caught": ""})
        print(f"{rate:16.2%}   {rate * WINDOW_SECONDS:15.0%}   {caught:13.0%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "strength_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    fig, (left, right) = plt.subplots(1, 2, figsize=(11.5, 4.5), sharey=True)
    offsets = [r for r in rows if r["kind"] == "offset"]
    x = [r["strength"] * 100 for r in offsets]
    left.plot(x, [r["under_caught"] * 100 for r in offsets], "o-", color="#1f4e79", label="under-reporting")
    left.plot(x, [r["over_caught"] * 100 for r in offsets], "s--", color="#d1495b", label="over-reporting")
    left.set_xscale("log")
    left.set_xticks([2, 5, 10, 20, 50])
    left.set_xticklabels(["2", "5", "10", "20", "50"])
    left.set_xlabel("offset attack size (% of true speed)")
    left.set_ylabel("attacked frames that raise an alarm (%)")
    left.set_title("Constant offset", fontsize=11)
    left.legend(frameon=False)

    drifts = [r for r in rows if r["kind"] == "drift"]
    right.plot([r["strength"] * 100 for r in drifts], [r["under_caught"] * 100 for r in drifts],
               "o-", color="#1f4e79")
    right.set_xscale("log")
    right.set_xticks([0.25, 0.5, 1, 2, 5, 10])
    right.set_xticklabels(["0.25", "0.5", "1", "2", "5", "10"])
    right.set_xlabel(f"drift rate (% per second, over {WINDOW_SECONDS:.0f} s)")
    right.set_title("Slow drift", fontsize=11)

    for ax in (left, right):
        ax.axhline(false_alarms * 100, color="black", lw=0.8, ls="--")
        ax.grid(alpha=0.25)
        ax.spines[["top", "right"]].set_visible(False)
    left.text(2, false_alarms * 100 + 2, "false alarm rate on clean frames", fontsize=8)
    fig.suptitle(f"Baseline threshold detector on {len(used)} held-out clear daytime scenes", fontsize=11)
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "strength_sweep.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()

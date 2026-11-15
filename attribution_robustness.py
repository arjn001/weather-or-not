"""
Step 4 check: do the attribution numbers hold up across different splits?

    python attribution_robustness.py

attribution_eval.py reports one train/test split. A single split can be
lucky or unlucky, depending on which scenes happen to land in the test
half. This script repeats the whole experiment several times, each time
shuffling the scenes into a fresh train half and test half (always by whole
scene, within each weather condition), and reports the average and the
spread of the headline numbers.

If the spread is small next to the gaps between methods, the conclusions
are safe to state. If it is large, they are not.

Writes out/attribution_robustness.csv.
"""

import csv
import os

import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from sklearn.ensemble import HistGradientBoostingClassifier

from attribution_features import build_rows, clear_reference, feature_names, to_matrix
from calibration import estimate_speed_scale
from scene_io import DATA_ROOT
from scene_sets import usable_scenes

OUT_DIR = "out"
N_REPEATS = 8
BASELINE_QUANTILE = 0.95


def random_split(scenes, rng):
    """Shuffle scenes within each condition and cut each group in half."""
    train, test = [], []
    for condition in sorted({s["condition"] for s in scenes}):
        group = [s for s in scenes if s["condition"] == condition]
        order = rng.permutation(len(group))
        half = (len(group) + 1) // 2
        train += [group[i] for i in order[:half]]
        test += [group[i] for i in order[half:]]
    return train, test


def share(mask, flagged):
    return float(np.mean(flagged[mask])) if mask.any() else np.nan


def one_repeat(nusc_can, scenes, seed):
    rng = np.random.default_rng(seed)
    train_scenes, test_scenes = random_split(scenes, rng)
    train_clear = [s for s in train_scenes if s["condition"] == "clear"]
    speed_scale, _ = estimate_speed_scale(nusc_can, train_clear)
    reference = clear_reference([s["name"] for s in train_clear])
    train = build_rows(nusc_can, train_scenes, speed_scale, reference)
    test = build_rows(nusc_can, test_scenes, speed_scale, reference)

    y_test = np.array([r["label"] for r in test])
    is_attack, is_degraded, is_nominal = (y_test == c for c in ("attack", "degraded", "nominal"))
    at_night = np.array([r["condition"].startswith("night") for r in test])
    in_clear = np.array([r["condition"] == "clear" for r in test])
    result = {"seed": seed}

    # The two classifiers.
    for tag, names in (("residual", feature_names(False)), ("full", feature_names(True))):
        X_train, y_train = to_matrix(train, names)
        X_test, _ = to_matrix(test, names)
        model = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
        called_attack = model.fit(X_train, y_train).predict(X_test) == "attack"
        result[f"{tag}_degraded_called_attack"] = share(is_degraded, called_attack)
        result[f"{tag}_nominal_called_attack"] = share(is_nominal, called_attack)
        result[f"{tag}_attacks_caught"] = share(is_attack, called_attack)
        result[f"{tag}_attacks_caught_clear"] = share(is_attack & in_clear, called_attack)
        result[f"{tag}_attacks_caught_night"] = share(is_attack & at_night, called_attack)

    # The threshold baseline, raised until it catches as many attacks as
    # the full classifier, so the comparison is like for like.
    nominal_train = [r for r in train if r["label"] == "nominal"]
    thr_speed = np.quantile([r["res_speed_abs_median"] for r in nominal_train], BASELINE_QUANTILE)
    thr_yaw = np.quantile([r["res_yaw_abs_median"] for r in nominal_train], BASELINE_QUANTILE)
    score = np.array([max(r["res_speed_abs_median"] / thr_speed, r["res_yaw_abs_median"] / thr_yaw)
                      for r in test])
    cut = np.quantile(score[is_attack], 1.0 - result["full_attacks_caught"])
    flagged = score > cut
    result["baseline_degraded_called_attack"] = share(is_degraded, flagged)
    result["baseline_nominal_called_attack"] = share(is_nominal, flagged)
    return result


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    scenes = usable_scenes(nusc_can)
    results = [one_repeat(nusc_can, scenes, seed) for seed in range(N_REPEATS)]

    print(f"{N_REPEATS} random train/test splits of {len(scenes)} scenes\n")
    print("measure                                    mean    spread   lowest  highest")
    keys = [k for k in results[0] if k != "seed"]
    for key in keys:
        values = np.array([r[key] for r in results])
        print(f"{key:40}  {values.mean():6.1%}  +/-{values.std():5.1%}  "
              f"{values.min():6.1%}  {values.max():6.1%}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "attribution_robustness.csv")
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""
Step 4, third design: the gate first, then a learned detector.

    python combined_eval.py

verify_gate_eval.py showed that an image-only gate handles darkness well
but leaves rain false alarms almost untouched. attribution_eval.py showed
that a classifier looking at HOW the camera and CAN disagree (steadily, or
noisily) handles rain well. This script puts the two together:

  1. the gate decides, from the image alone, whether to check a window
  2. inside checked windows, a classifier that sees only the disagreement
     statistics decides "attack" or "no attack"

The classifier in step 2 is deliberately given no image measurements. If it
had them it could learn "the image looks poor, so forgive this", which is
the loophole the gate exists to close.

For a like-for-like comparison the plain threshold detector is also run
inside the same checked windows, and both detectors are set to catch the
same share of attacks before their false alarms are compared.

A second table shows what happens as the gate is made stricter.

Repeated over several random train/test splits by whole scene.
Writes out/combined.csv.
"""

import csv
import os

import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from sklearn.ensemble import HistGradientBoostingClassifier

from attribution_features import build_rows, clear_reference, feature_names, to_matrix
from attribution_robustness import random_split
from calibration import estimate_speed_scale
from scene_io import DATA_ROOT
from scene_sets import usable_scenes
from verify_gate_eval import BASELINE_QUANTILE, GATE_CONFIDENCE, GATE_FEATURES, GROUPS, group_of, share

OUT_DIR = "out"
N_REPEATS = 8
CONFIDENCE_LEVELS = (0.3, 0.5, 0.7, 0.9)


def one_repeat(nusc_can, scenes, seed):
    rng = np.random.default_rng(seed)
    train_scenes, test_scenes = random_split(scenes, rng)
    train_clear = [s for s in train_scenes if s["condition"] == "clear"]
    speed_scale, _ = estimate_speed_scale(nusc_can, train_clear)
    reference = clear_reference([s["name"] for s in train_clear])
    train = build_rows(nusc_can, train_scenes, speed_scale, reference)
    test = build_rows(nusc_can, test_scenes, speed_scale, reference)

    nominal_train = [r for r in train if r["label"] == "nominal"]
    thr_speed = np.quantile([r["res_speed_abs_median"] for r in nominal_train], BASELINE_QUANTILE)
    thr_yaw = np.quantile([r["res_yaw_abs_median"] for r in nominal_train], BASELINE_QUANTILE)

    def threshold_score(rows):
        """How many times over its threshold a window is. Above 1 is an alarm."""
        return np.array([max(r["res_speed_abs_median"] / thr_speed, r["res_yaw_abs_median"] / thr_yaw)
                         for r in rows])

    # The gate: trained on clean windows, image measurements only.
    clean_train = [r for r in train if r["label"] != "attack"]
    gate = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    gate.fit(to_matrix(clean_train, GATE_FEATURES)[0], threshold_score(clean_train) <= 1.0)

    def gate_confidence(rows):
        return gate.predict_proba(to_matrix(rows, GATE_FEATURES)[0])[:, list(gate.classes_).index(True)]

    # The detector: disagreement statistics only, trained on the training
    # windows the gate would have let through, since that is all it will
    # ever be asked about.
    names = feature_names(include_health=False)
    passed_train = [r for r, c in zip(train, gate_confidence(train)) if c >= GATE_CONFIDENCE]
    X_train, y_train = to_matrix(passed_train, names)
    detector = HistGradientBoostingClassifier(random_state=seed, class_weight="balanced")
    detector.fit(X_train, y_train == "attack")

    X_test, y_test = to_matrix(test, names)
    is_attack = y_test == "attack"
    confidence = gate_confidence(test)
    checked = confidence >= GATE_CONFIDENCE
    groups = np.array([group_of(r["condition"]) for r in test])
    score = threshold_score(test)
    learned_alarm = detector.predict(X_test)

    # Raise the threshold until it catches as many attacks in checked
    # windows as the learned detector does.
    caught = share(is_attack & checked, learned_alarm)
    cut = np.quantile(score[is_attack & checked], 1.0 - caught)
    matched_alarm = score > cut

    # The learned detector gives each window a probability of attack. Its
    # default cut-off favours catching attacks. To compare fairly with the
    # threshold detector we also set both to catch exactly 90 percent of
    # attacks in checked windows and see how often each one cries wolf.
    attack_probability = detector.predict_proba(X_test)[:, list(detector.classes_).index(True)]
    learned_at_90 = attack_probability > np.quantile(attack_probability[is_attack & checked], 0.10)
    threshold_at_90 = score > np.quantile(score[is_attack & checked], 0.10)

    result = {"seed": seed}
    for g in GROUPS + ("all",):
        here = np.ones(len(test), bool) if g == "all" else groups == g
        clean, attacked = here & ~is_attack & checked, here & is_attack & checked
        result[f"{g}: false alarms at 90% catch rate, threshold"] = share(clean, threshold_at_90)
        result[f"{g}: false alarms at 90% catch rate, learned detector"] = share(clean, learned_at_90)
        result[f"{g}: clean windows checked"] = share(here & ~is_attack, checked)
        result[f"{g}: false alarms, threshold"] = share(clean, score > 1.0)
        result[f"{g}: false alarms, threshold at matched catch rate"] = share(clean, matched_alarm)
        result[f"{g}: false alarms, learned detector"] = share(clean, learned_alarm)
        result[f"{g}: attacks caught, threshold"] = share(attacked, score > 1.0)
        result[f"{g}: attacks caught, learned detector"] = share(attacked, learned_alarm)

    # Stricter and looser gates, with the plain threshold behind them.
    for level in CONFIDENCE_LEVELS:
        passed = confidence >= level
        for g in GROUPS:
            clean = (groups == g) & ~is_attack
            result[f"gate {level}: {g} checked"] = share(clean, passed)
            result[f"gate {level}: {g} false alarms"] = share(clean & passed, score > 1.0)
    return result


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    scenes = usable_scenes(nusc_can)
    results = [one_repeat(nusc_can, scenes, seed) for seed in range(N_REPEATS)]
    keys = [k for k in results[0] if k != "seed"]
    mean = {k: float(np.nanmean([r[k] for r in results])) for k in keys}
    spread = {k: float(np.nanstd([r[k] for r in results])) for k in keys}

    print(f"{N_REPEATS} random train/test splits of {len(scenes)} scenes\n")
    print("INSIDE CHECKED WINDOWS (gate confidence 0.5)")
    print(f"{'':50}" + "".join(f"{g:>16}" for g in GROUPS + ("all",)))
    for m in [k.split(": ", 1)[1] for k in keys if k.startswith("clear: ")]:
        cells = "".join(f"{mean[f'{g}: {m}']:9.1%} +/-{spread[f'{g}: {m}']:4.1%}" for g in GROUPS + ("all",))
        print(f"{m:50}{cells}")

    print("\nSTRICTER OR LOOSER GATE (plain threshold behind it)")
    print("gate confidence   " + "".join(f"{g + ' checked':>15}{g + ' false al.':>17}" for g in GROUPS))
    for level in CONFIDENCE_LEVELS:
        cells = "".join(f"{mean[f'gate {level}: {g} checked']:15.1%}{mean[f'gate {level}: {g} false alarms']:17.1%}"
                        for g in GROUPS)
        print(f"{level:<18}{cells}")

    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, "combined.csv")
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["measure", "mean", "spread"])
        for k in keys:
            writer.writerow([k, mean[k], spread[k]])
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()

"""
Step 4: can we tell an attack apart from a struggling camera?

    python attribution_eval.py

The baseline detector raises an alarm whenever CAN and camera disagree.
Rain and night make the camera worse, so they raise the same alarm. Here a
classifier looks at a ~2 second window and picks one of three answers:

  nominal  : clear weather, nothing wrong
  degraded : camera struggling (rain, night, wet road), no attack
  attack   : CAN data falsified

Two classifiers are trained so we can see what the camera health signals
add: one that sees only the residual features, and one that also sees the
health features. A plain threshold on the residual (the baseline) is scored
on the same test windows for comparison.

Train and test never share a scene. Each condition is split by whole scene
with calibration.split_scenes, because windows from one scene are near
copies of each other. The speed scale and the clear-weather reference are
measured on the clear TRAINING scenes only.

Writes out/attribution_results.csv and out/attribution_confusion.png.
"""

import csv
import os
from collections import Counter

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from nuscenes.can_bus.can_bus_api import NuScenesCanBus
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

from attribution_features import build_rows, clear_reference, feature_names, to_matrix
from calibration import estimate_speed_scale, split_scenes
from scene_io import DATA_ROOT
from scene_sets import usable_scenes

OUT_DIR = "out"
CLASSES = ["nominal", "degraded", "attack"]
DEGRADED_CONDITIONS = ["rain", "night", "night+rain", "wet"]

# The baseline threshold is the level that this share of clean nominal
# TRAINING windows stay below.
BASELINE_QUANTILE = 0.95


def split_all(scenes):
    """
    Split every condition by whole scene into train and test.

    A condition with fewer than 2 scenes cannot be split, so it goes to
    train only. The names of those conditions are returned so the output
    can say so.
    """
    train, test, train_only = [], [], []
    for condition in ["clear"] + DEGRADED_CONDITIONS:
        group = [s for s in scenes if s["condition"] == condition]
        if len(group) == 0:
            continue
        if len(group) < 2:
            train.extend(group)
            train_only.append(condition)
            continue
        first, second = split_scenes(group)
        train.extend(first)
        test.extend(second)
    return train, test, train_only


def share(flags):
    """Share of True in a boolean array, NaN if the array is empty."""
    flags = np.asarray(flags)
    return float(flags.mean()) if len(flags) else np.nan


def pct(x):
    return "   n/a" if not np.isfinite(x) else f"{100 * x:5.1f}%"


def headline_shares(y_true, y_flagged_attack):
    """Share of each true class that was called attack."""
    out = {}
    for label in CLASSES:
        out[label] = share(y_flagged_attack[y_true == label])
    return out


def print_confusion(cm):
    print("  confusion matrix, counts (rows = true, columns = predicted)")
    print("  " + " " * 10 + "".join(f"{c:>10}" for c in CLASSES))
    for i, c in enumerate(CLASSES):
        print("  " + f"{c:<10}" + "".join(f"{v:10d}" for v in cm[i]))
    print("  row percentages")
    for i, c in enumerate(CLASSES):
        total = cm[i].sum()
        cells = "".join(f"{(100 * v / total if total else np.nan):9.1f}%" for v in cm[i])
        print("  " + f"{c:<10}" + cells)


def row_normalise(cm):
    totals = cm.sum(axis=1, keepdims=True)
    return np.divide(cm, totals, out=np.full(cm.shape, np.nan), where=totals > 0)


def attack_recall_breakdown(rows_test, called_attack):
    """Print attack recall by injected scene condition and by attack name."""
    is_attack = np.array([r["label"] == "attack" for r in rows_test])
    for title, key in [("by scene condition", "condition"), ("by attack name", "attack")]:
        print(f"  attack recall {title}")
        groups = sorted({r[key] for r, a in zip(rows_test, is_attack) if a})
        if not groups:
            print("    (no attack windows in test)")
        for g in groups:
            sel = np.array([a and r[key] == g for r, a in zip(rows_test, is_attack)])
            print(f"    {g:<16} {pct(share(called_attack[sel]))}  ({sel.sum()} windows)")


def fit_and_report(title, names, rows_train, rows_test):
    """Train one classifier, print its report, return results for saving."""
    X_train, y_train = to_matrix(rows_train, names)
    X_test, y_test = to_matrix(rows_test, names)
    print(f"\n=== {title} ({len(names)} features) ===")
    if len(set(y_train)) < 2 or len(y_test) == 0:
        print("  not enough classes in train or any test rows, skipped")
        return None

    model = HistGradientBoostingClassifier(random_state=0, class_weight="balanced")
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

    cm = confusion_matrix(y_test, pred, labels=CLASSES)
    print_confusion(cm)
    present = [c for c in CLASSES if (y_test == c).any()]
    bal = float(balanced_accuracy_score(y_test, pred)) if present else np.nan
    print(f"  balanced accuracy: {pct(bal)}")

    called_attack = pred == "attack"
    heads = headline_shares(y_test, called_attack)
    print(f"  degraded windows called attack: {pct(heads['degraded'])}")
    print(f"  nominal windows called attack:  {pct(heads['nominal'])}")
    print(f"  attack windows called attack:   {pct(heads['attack'])}")
    attack_recall_breakdown(rows_test, called_attack)
    return {"model": model, "cm": cm, "heads": heads, "balanced_accuracy": bal,
            "X_test": X_test, "y_test": y_test}


def main():
    nusc_can = NuScenesCanBus(dataroot=DATA_ROOT)
    scenes = usable_scenes(nusc_can)
    train_scenes, test_scenes, train_only = split_all(scenes)

    print("scene counts (train / test)")
    for condition in ["clear"] + DEGRADED_CONDITIONS:
        n_train = sum(s["condition"] == condition for s in train_scenes)
        n_test = sum(s["condition"] == condition for s in test_scenes)
        note = "  (fewer than 2 scenes, train only)" if condition in train_only else ""
        print(f"  {condition:<11} {n_train:4d} / {n_test:4d}{note}")

    train_clear = [s for s in train_scenes if s["condition"] == "clear"]
    if not train_clear:
        raise SystemExit("no clear training scenes, cannot calibrate")
    speed_scale, ratios = estimate_speed_scale(nusc_can, train_clear)
    reference = clear_reference([s["name"] for s in train_clear])
    print(f"\nspeed scale from {len(ratios)} clear training scenes: {speed_scale:.3f}")

    rows_train = build_rows(nusc_can, train_scenes, speed_scale, reference)
    rows_test = build_rows(nusc_can, test_scenes, speed_scale, reference)
    for name, rows in [("train", rows_train), ("test", rows_test)]:
        counts = Counter(r["label"] for r in rows)
        print(f"{name} windows: " + ", ".join(f"{c} {counts.get(c, 0)}" for c in CLASSES))

    # Baseline: a threshold on each residual channel, set on nominal train.
    y_test = np.array([r["label"] for r in rows_test])
    nominal_train = [r for r in rows_train if r["label"] == "nominal"]
    print("\n=== Baseline threshold detector ===")
    if not nominal_train or len(rows_test) == 0:
        print("  no nominal training windows or no test windows, skipped")
        base_heads = {c: np.nan for c in CLASSES}
    else:
        thr_speed = float(np.quantile([r["res_speed_abs_median"] for r in nominal_train], BASELINE_QUANTILE))
        thr_yaw = float(np.quantile([r["res_yaw_abs_median"] for r in nominal_train], BASELINE_QUANTILE))
        print(f"  thresholds (95th percentile of nominal train): speed {thr_speed:.3f}, yaw {thr_yaw:.3f} rad/s")
        flagged = np.array([(r["res_speed_abs_median"] > thr_speed) or (r["res_yaw_abs_median"] > thr_yaw)
                            for r in rows_test])
        base_heads = headline_shares(y_test, flagged)
        print(f"  nominal windows flagged:  {pct(base_heads['nominal'])}")
        print(f"  degraded windows flagged: {pct(base_heads['degraded'])}")
        print(f"  attack windows flagged:   {pct(base_heads['attack'])}")

    resid = fit_and_report("Residual features only", feature_names(False), rows_train, rows_test)
    full = fit_and_report("Residual + camera health", feature_names(True), rows_train, rows_test)

    # A fair comparison. The baseline above catches more attacks than the
    # classifier does, simply because its threshold sits lower. So we also
    # raise the baseline's thresholds, both by the same factor, until it
    # catches exactly as many attacks as the full classifier. Then the only
    # difference left between the two is how often they cry wolf.
    matched_heads = None
    if full is not None and nominal_train:
        score = np.array([max(r["res_speed_abs_median"] / thr_speed, r["res_yaw_abs_median"] / thr_yaw)
                          for r in rows_test])
        target = full["heads"]["attack"]
        cut = float(np.quantile(score[y_test == "attack"], 1.0 - target))
        matched_heads = headline_shares(y_test, score > cut)
        print(f"\n=== Baseline with thresholds raised {cut:.2f}x to match the classifier's attack recall ===")
        print(f"  nominal windows flagged:  {pct(matched_heads['nominal'])}")
        print(f"  degraded windows flagged: {pct(matched_heads['degraded'])}")
        print(f"  attack windows flagged:   {pct(matched_heads['attack'])}")

    if full is not None:
        print("\n=== Permutation importance, full model, test rows ===")
        print("  (drop in balanced accuracy when one feature is shuffled)")
        imp = permutation_importance(full["model"], full["X_test"], full["y_test"], n_repeats=5,
                                     random_state=0, scoring="balanced_accuracy")
        order = np.argsort(imp.importances_mean)[::-1][:10]
        names = feature_names(True)
        for i in order:
            print(f"  {names[i]:<28} {imp.importances_mean[i]:7.4f} +/- {imp.importances_std[i]:.4f}")

    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, "attribution_results.csv")
    fields = ["method", "degraded_called_attack", "nominal_called_attack",
              "attack_called_attack", "balanced_accuracy"]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow({"method": "baseline threshold", "degraded_called_attack": base_heads["degraded"],
                         "nominal_called_attack": base_heads["nominal"],
                         "attack_called_attack": base_heads["attack"], "balanced_accuracy": ""})
        if matched_heads is not None:
            writer.writerow({"method": "baseline threshold, recall matched to full classifier",
                             "degraded_called_attack": matched_heads["degraded"],
                             "nominal_called_attack": matched_heads["nominal"],
                             "attack_called_attack": matched_heads["attack"], "balanced_accuracy": ""})
        for method, result in [("residual only", resid), ("residual + health", full)]:
            if result is None:
                continue
            writer.writerow({"method": method, "degraded_called_attack": result["heads"]["degraded"],
                             "nominal_called_attack": result["heads"]["nominal"],
                             "attack_called_attack": result["heads"]["attack"],
                             "balanced_accuracy": result["balanced_accuracy"]})

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, title, result in zip(axes, ["Residual features only", "Residual + camera health"], [resid, full]):
        ax.set_title(title, fontsize=11)
        if result is None:
            ax.axis("off")
            continue
        norm = row_normalise(result["cm"])
        ax.imshow(np.nan_to_num(norm), vmin=0, vmax=1, cmap="Blues")
        for i in range(3):
            for j in range(3):
                text = "n/a" if not np.isfinite(norm[i, j]) else f"{100 * norm[i, j]:.0f}%"
                ax.text(j, i, text, ha="center", va="center",
                        color="white" if np.nan_to_num(norm[i, j]) > 0.5 else "black")
        ax.set_xticks(range(3), CLASSES)
        ax.set_yticks(range(3), CLASSES)
        ax.set_xlabel("predicted")
        ax.set_ylabel("true")
    fig.tight_layout()
    fig_path = os.path.join(OUT_DIR, "attribution_confusion.png")
    fig.savefig(fig_path, dpi=130)
    plt.close(fig)
    print(f"\nwrote {csv_path}\nwrote {fig_path}")


if __name__ == "__main__":
    main()

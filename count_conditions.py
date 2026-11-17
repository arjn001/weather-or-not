"""
How many scenes are rain, night, or clear?

    python count_conditions.py                  # full dataset (v1.0-trainval)
    python count_conditions.py v1.0-mini        # the 10 mini scenes

Every nuScenes scene has a short description written by a person, such as
"Night, after rain, many peds". This reads those descriptions and sorts the
scenes into groups by which words appear. It needs only the small metadata
download, no images.

Writes out/scene_conditions_<version>.csv with one row per scene.

The answer caps the whole benign-degradation side of the project: we cannot
test on more real rain and night scenes than exist.
"""

import csv
import json
import os
import re
import sys

DATA_ROOT = "data/nuscenes"
OUT_DIR = "out"


def has_word(text, word):
    """True if `word` appears as a whole word, so 'rain' does not match 'train'."""
    return re.search(rf"\b{word}", text) is not None


def classify(description):
    """
    Sort one description into flags.

    'after rain' and 'wet' mean a wet road with nothing falling. That is a
    different kind of degradation from rain on the lens, so it gets its own
    flag instead of being lumped in with rain.
    """
    text = description.lower()
    night = has_word(text, "night")
    rain = has_word(text, "rain") and "after rain" not in text
    wet_only = ("after rain" in text or has_word(text, "wet")) and not rain
    return {"night": night, "rain": rain, "wet_road": wet_only}


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v1.0-trainval"
    path = os.path.join(DATA_ROOT, version, "scene.json")
    if not os.path.exists(path):
        sys.exit(f"Cannot find {path}. Download and extract the {version} metadata first.")

    with open(path) as f:
        scenes = json.load(f)

    rows = []
    for scene in sorted(scenes, key=lambda s: s["name"]):
        flags = classify(scene["description"])
        rows.append({"scene": scene["name"], **flags, "description": scene["description"]})

    total = len(rows)
    night = sum(r["night"] for r in rows)
    rain = sum(r["rain"] for r in rows)
    both = sum(r["night"] and r["rain"] for r in rows)
    wet = sum(r["wet_road"] for r in rows)
    clear_day = sum(not (r["night"] or r["rain"] or r["wet_road"]) for r in rows)

    print(f"{version}: {total} scenes")
    print(f"  night              {night:4d}")
    print(f"  rain               {rain:4d}")
    print(f"  night AND rain     {both:4d}")
    print(f"  wet road, no rain  {wet:4d}")
    print(f"  clear daytime      {clear_day:4d}")

    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"scene_conditions_{version}.csv")
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()

"""
Build a small per-scene index of the front camera frames.

    python build_index.py                # v1.0-trainval, all 850 scenes
    python build_index.py v1.0-mini      # the 10 mini scenes

Why this exists. The nuScenes devkit loads every table of the dataset into
memory before it lets you ask anything. For mini that is instant. For the
full dataset the tables are about 2.5 GB of text and loading them takes
minutes and most of a laptop's memory, and we would pay that every time any
script starts.

We only ever need three things per scene: the list of front camera frames
with their timestamps, the camera calibration, and the description. So this
script reads the big tables once, pulls those out, and writes one small
file per scene to index/. Everything else in the project reads the index.

The biggest table (sample_data.json, 1.3 GB) is read one record at a time
instead of all at once, so this runs in a few hundred MB of memory.

It only needs the metadata. It does not care whether the images have been
downloaded yet.
"""

import json
import os
import sys
from collections import defaultdict

from scene_io import camera_geometry

DATA_ROOT = "data/nuscenes"
INDEX_DIR = "index"
CAMERA = "CAM_FRONT"


def read_table(version, name):
    with open(os.path.join(DATA_ROOT, version, f"{name}.json")) as f:
        return json.load(f)


def front_camera_records(version):
    """
    Yield the sample_data records that belong to the front camera.

    The file is one long list with each record spread over several lines
    between a "{" line and a "}" line. We collect the lines of one record
    at a time. Most records are lidar, radar or other cameras, so we check
    for the camera name in the raw text first and only decode the ones we
    want. That skips about 11 records out of every 12.
    """
    wanted = f"/{CAMERA}/"
    path = os.path.join(DATA_ROOT, version, "sample_data.json")
    lines, is_wanted = [], False
    with open(path) as f:
        for line in f:
            stripped = line.strip()
            if stripped == "{":
                lines, is_wanted = ["{"], False
            elif stripped in ("}", "},"):
                if is_wanted:
                    lines.append("}")
                    yield json.loads("".join(lines))
            else:
                lines.append(stripped)
                if wanted in stripped:
                    is_wanted = True


def main():
    version = sys.argv[1] if len(sys.argv) > 1 else "v1.0-trainval"

    scenes = {s["token"]: s for s in read_table(version, "scene")}
    scene_of_sample = {s["token"]: s["scene_token"] for s in read_table(version, "sample")}
    calibrations = {c["token"]: c for c in read_table(version, "calibrated_sensor")}

    # Group the front camera frames by the scene they belong to.
    frames = defaultdict(list)
    for record in front_camera_records(version):
        scene_token = scene_of_sample[record["sample_token"]]
        frames[scene_token].append(
            (record["timestamp"], record["filename"], record["calibrated_sensor_token"])
        )

    os.makedirs(INDEX_DIR, exist_ok=True)
    for token, scene in scenes.items():
        scene_frames = sorted(frames[token])
        if not scene_frames:
            print(f"  {scene['name']}: no front camera frames, skipped")
            continue
        camera = camera_geometry(calibrations[scene_frames[0][2]])
        entry = {
            "name": scene["name"],
            "version": version,
            "description": scene["description"],
            "utimes": [f[0] for f in scene_frames],
            "files": [f[1] for f in scene_frames],
            # numpy arrays cannot be written to JSON, plain lists can
            "camera": {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in camera.items()},
        }
        with open(os.path.join(INDEX_DIR, f"{scene['name']}.json"), "w") as f:
            json.dump(entry, f)

    print(f"{version}: wrote {len(frames)} scene files to {INDEX_DIR}/")


if __name__ == "__main__":
    main()

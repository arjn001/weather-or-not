"""
Checks on scene selection, the index reader and the image measurements.

Like test_core.py these use made-up inputs and do not need the dataset.
"""

import json
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import build_index
from camera_health import frame_appearance
from count_conditions import classify
from scene_sets import can_problem, condition_of


# ----------------------------------------------------------- condition labels

def test_rain_is_not_found_inside_other_words():
    assert classify("Train tracks, bus, restraint area")["rain"] is False


def test_after_rain_is_a_wet_road_not_rain():
    flags = classify("Night, after rain, many peds")
    assert flags == {"night": True, "rain": False, "wet_road": True}


def test_wet_ground_while_raining_counts_as_rain_only():
    flags = classify("Rain, wet ground, bendy bus")
    assert flags == {"night": False, "rain": True, "wet_road": False}


def test_condition_labels():
    assert condition_of("Parked truck, construction") == "clear"
    assert condition_of("Rain, parking lot") == "rain"
    assert condition_of("Night, big street") == "night"
    assert condition_of("Night, rain, bus") == "night+rain"
    assert condition_of("Wet road, turn left") == "wet"


# ------------------------------------------------------------------ CAN check

class FakeCanBus:
    """Stands in for the devkit's CAN reader."""

    def __init__(self, messages, blacklist=()):
        self.messages = messages
        self.can_blacklist = list(blacklist)

    def get_messages(self, scene_name, message_name):
        return self.messages


def pose_messages(seconds=20, hz=50, skip=()):
    out = []
    for i in range(seconds * hz):
        if i in skip:
            continue
        out.append({"utime": int(i * 1e6 / hz), "vel": [5.0, 0, 0], "rotation_rate": [0, 0, 0.0]})
    return out


def scene_of_length(seconds=20, name="scene-0001"):
    return {"name": name, "utimes": np.array([0, int(seconds * 1e6)], dtype=np.int64)}


def test_healthy_can_passes():
    assert can_problem(FakeCanBus(pose_messages()), scene_of_length()) is None


def test_blacklisted_scene_is_rejected():
    bus = FakeCanBus(pose_messages(), blacklist=[161])
    assert can_problem(bus, scene_of_length(name="scene-0161")) is not None


def test_can_that_stops_early_is_rejected():
    bus = FakeCanBus(pose_messages(seconds=12))
    assert can_problem(bus, scene_of_length(seconds=20)) is not None


def test_gap_in_the_middle_is_rejected():
    hole = set(range(400, 460))  # 1.2 seconds missing, too few to fail the count
    assert can_problem(FakeCanBus(pose_messages(skip=hole)), scene_of_length()) is not None


# --------------------------------------------------------- image measurements

def test_blur_lowers_sharpness_and_darkening_lowers_brightness():
    rng = np.random.default_rng(0)
    crisp = (rng.random((200, 300)) * 255).astype(np.uint8)
    blurred = cv2.GaussianBlur(crisp, (0, 0), 3)
    dark = (crisp * 0.2).astype(np.uint8)

    a, b, d = frame_appearance(crisp), frame_appearance(blurred), frame_appearance(dark)
    assert b["sharpness"] < 0.2 * a["sharpness"]
    assert d["brightness"] < 0.3 * a["brightness"]
    assert d["contrast"] < 0.3 * a["contrast"]


# --------------------------------------------------------------- index reader

def test_index_reader_keeps_only_front_camera_records(tmp_path, monkeypatch):
    records = [
        {"token": "a", "filename": "samples/CAM_FRONT/x.jpg", "timestamp": 2},
        {"token": "b", "filename": "samples/CAM_FRONT_LEFT/y.jpg", "timestamp": 1},
        {"token": "c", "filename": "sweeps/LIDAR_TOP/z.bin", "timestamp": 3},
        {"token": "d", "filename": "sweeps/CAM_FRONT/w.jpg", "timestamp": 4},
    ]
    # Write the file the way nuScenes does: one field per line.
    folder = tmp_path / "v-test"
    folder.mkdir()
    (folder / "sample_data.json").write_text(json.dumps(records, indent=0))

    monkeypatch.setattr(build_index, "DATA_ROOT", str(tmp_path))
    found = list(build_index.front_camera_records("v-test"))
    assert [r["token"] for r in found] == ["a", "d"]

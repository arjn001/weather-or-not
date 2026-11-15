# Weather or Not

Distinguishing sensor attacks from benign degradation in vehicle camera and
CAN streams.

CS 4337 Computer Vision, Fall 2026, Texas State University.

## The idea

A car can measure its own motion two ways: from the CAN bus (wheel sensors
and an inertial unit) and from a forward-facing camera. When the two
disagree, something is wrong. Existing work uses that disagreement to catch
falsified CAN data, but it assumes the camera works. Rain and darkness break
the camera and raise the same alarm.

This project measures how badly that fails on real driving data, and builds
a detector that knows when the camera is in no state to check the CAN bus.

## What is in here

The pipeline has five stages. Each stage is one or two small files.

| stage | what it does | files |
|---|---|---|
| 1. Camera motion | Turn rate from tracked corner points (Lucas-Kanade optical flow and an essential matrix). Forward speed from matching the road surface between frames. | `camera_motion.py`, `camera_health.py` |
| 2. Compare with CAN | Bring CAN onto the camera's timeline, then compute how far the two disagree (the "residual") for speed and for turn rate. | `scene_io.py`, `residuals.py`, `calibration.py` |
| 3. Attacks | Falsify the CAN stream in three ways: constant offset, slow drift, replay of earlier data. | `attacks.py` |
| 4. Baseline detector | A threshold on the smoothed residual. This is the existing approach. | `detector.py`, `baseline_eval.py`, `strength_sweep.py` |
| 5. Attribution | Decide whether a disagreement is an attack, bad conditions, or cannot be checked at all. | `attribution_features.py`, `attribution_eval.py`, `verify_gate_eval.py`, `combined_eval.py` |

Supporting files:

| file | purpose |
|---|---|
| `build_index.py` | Reads the dataset's metadata once and writes a small per-scene index. |
| `fetch_part.sh` | Downloads one part of the dataset, keeping only the front camera. |
| `process_scenes.py` | Runs stage 1 on every downloaded scene and caches the result. |
| `camera_cache.py` | Loads and saves the cached camera results. |
| `scene_sets.py` | Labels each scene (clear, rain, night) and rejects scenes with bad CAN data. |
| `count_conditions.py` | Counts rain, night and clear scenes from the scene descriptions. |
| `kill_gate.py` | Plots camera against CAN for a scene. The first sanity check. |
| `residual_report.py` | Noise floor table and an example figure of the three attacks. |
| `false_alarm_eval.py` | Runs the baseline in rain and at night with no attack present. |
| `attribution_robustness.py` | Repeats the attribution experiment over 8 random splits. |
| `check_data.py` | Confirms the dataset and CAN expansion are readable. |
| `tests/` | 44 quick checks on made-up signals. No dataset needed. |

`PROJECT_BRIEF.md` is the original plan. `PROGRESS.md` is the lab notebook:
every experiment, its numbers, and what did not work.

## Data

[nuScenes](https://www.nuscenes.org) with its CAN bus expansion. Front
camera at about 12 frames per second, CAN pose messages at 50 per second,
scenes of 20 seconds each.

The data is not in this repository. nuScenes is free for non-commercial use
and must be downloaded by each user. The experiments here use 259 scenes:
118 clear daytime, 56 rain, 70 night, 15 night with rain.

## Setup

```bash
conda create -n weather python=3.10
conda activate weather
pip install -r requirements.txt
python -m pytest                 # 44 tests, a few seconds, no data needed
```

## Getting the data

About 5 GB for the first step, then about 3.5 GB per dataset part.

```bash
mkdir -p data/nuscenes && cd data/nuscenes

# small set of 10 scenes, the CAN expansion, and the full metadata
curl -f -L -O https://www.nuscenes.org/data/v1.0-mini.tgz
curl -f -L -O https://motional-nuscenes.s3.amazonaws.com/public/v1.0/can_bus.zip
curl -f -L -O https://motional-nuscenes.s3.amazonaws.com/public/v1.0/v1.0-trainval_meta.tgz
tar -xzf v1.0-mini.tgz && unzip -q can_bus.zip && tar -xzf v1.0-trainval_meta.tgz
cd ../..

python check_data.py             # confirms everything is readable
python build_index.py            # writes index/, about 10 seconds

# full dataset parts, front camera only. Part 1 is clear weather,
# part 6 is mostly rain, part 10 is night.
bash fetch_part.sh 1
bash fetch_part.sh 6
bash fetch_part.sh 10

python process_scenes.py         # about 13 seconds per scene, writes cache/
```

After `process_scenes.py` has run, the images are no longer needed. Every
later script reads `cache/`.

## Running the experiments

Each script prints a table and writes a figure or CSV to `out/`.

| command | input | output |
|---|---|---|
| `python kill_gate.py scene-0061` | one cached scene | `out/kill_gate_scene-0061.png`: camera vs CAN speed and turn rate |
| `python count_conditions.py` | dataset metadata | `out/scene_conditions_v1.0-trainval.csv` |
| `python residual_report.py` | cached scenes | `out/clean_residuals.csv`, `out/attacks_scene-0103.png` |
| `python baseline_eval.py` | clear scenes | `out/baseline_auroc.csv`, `out/baseline_roc.png` |
| `python strength_sweep.py` | clear scenes | `out/strength_sweep.csv`, `out/strength_sweep.png` |
| `python false_alarm_eval.py` | all scenes | `out/false_alarms.csv`, `out/false_alarms.png` |
| `python attribution_eval.py` | all scenes | `out/attribution_results.csv`, `out/attribution_confusion.png` |
| `python attribution_robustness.py` | all scenes | `out/attribution_robustness.csv` |
| `python verify_gate_eval.py` | all scenes | `out/verify_gate.csv`, `out/verify_gate.png` |
| `python combined_eval.py` | all scenes | `out/combined.csv` |

Every evaluation splits the data by whole scene, so no scene is used both to
set a detector up and to test it.

## Results in brief

Full numbers and caveats are in `PROGRESS.md`.

- **The camera agrees with the CAN bus in clear daytime.** Speed correlation
  0.98 and turn rate correlation 0.99 on the reference scene. The camera
  reads speed about 4 percent high, which is measured and divided out.
- **The baseline works in clear weather for large attacks.** It catches
  about 88 percent of a 20 percent speed falsification. Attacks of 5 percent
  or less, or a drift of 1 percent per second or slower, are not detected.
- **It fails in bad conditions.** With no attack present it raises an alarm
  on 12 percent of clear frames, 23 percent of rain frames and 64 percent of
  night frames.
- **A gate that looks only at the image fixes most of that.** It checks 95
  percent of clear windows, 83 percent of rain windows and 27 percent of
  night windows, and answers "cannot verify" for the rest. Inside the
  windows it checks, a learned detector catches 90 percent of attacks with
  8 percent false alarms.
- **Open problem.** Rain false alarms stay near 13 percent, three times the
  clear rate. The image measurements used here do not predict which rain
  windows will fail.

## Folders not in git

| folder | contents | how to rebuild |
|---|---|---|
| `data/` | nuScenes images, CAN data, metadata | see "Getting the data" |
| `index/` | per-scene frame lists and calibration | `python build_index.py` |
| `cache/` | per-scene camera results | `python process_scenes.py` |

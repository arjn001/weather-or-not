# Weather or Not

Distinguishing sensor attacks from benign degradation in vehicle camera and
CAN streams.

CS 4337 Computer Vision, Fall 2026, Texas State University.

A car can measure its own motion two ways: from the CAN bus, and from a
forward-facing camera. When those disagree, something is wrong. Existing work
uses that to catch spoofed telemetry, but it assumes the camera works. Rain and
night break the camera and raise the same alarm. This project measures how
badly that fails, and asks whether the two causes can be told apart.

## Where to start

- `PROJECT_BRIEF.md` — what we are building and why. Read first.
- `PROGRESS.md` — what has actually been done. Read second.

## Running it

```bash
conda activate weather
python check_data.py            # confirms the data is readable
python kill_gate.py scene-0061  # camera vs CAN plot for one scene
python kill_gate.py all         # every scene, plus out/kill_gate.csv
```

Plots land in `out/`.

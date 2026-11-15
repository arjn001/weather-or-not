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
python -m pytest                # quick checks, a few seconds

python build_index.py           # once: small per-scene index from the metadata
bash fetch_part.sh 1            # download one dataset part, front camera only
python process_scenes.py        # process and cache every scene on disk

python kill_gate.py all         # camera vs CAN plot for every cached scene
python residual_report.py       # noise floor table and attack example figure
python baseline_eval.py         # AUROC of the threshold detector, clear weather
python false_alarm_eval.py      # Step 3: false alarms in rain and at night
python attribution_eval.py      # Step 4: nominal vs degraded vs attack classifier
python attribution_robustness.py  # Step 4 repeated over 8 random splits
python verify_gate_eval.py      # Step 4, second design: "cannot verify" gate
python combined_eval.py         # Step 4, third design: gate plus learned detector
```

Plots and tables land in `out/`. `data/`, `index/` and `cache/` are not in
git and are rebuilt with the commands above.

# Strict Python UltraTimTrack

This is the runnable Python GUI path for the current UltraTimTrack-style
pipeline.

## Run the GUI

```bash
python -m venv .venv
./.venv/bin/python -m pip install -r requirements
./.venv/bin/python scripts/strict_ultratimtrack_gui.py

if it doesn't work, try:

cd /Users/grosbedou/PycharmProjects/NDORMS
./.venv/bin/python scripts/strict_ultratimtrack_gui.py
```

The GUI writes each analysis to `results/strict_ultratimtrack_runs/<video-name>/`
with a CSV, compressed NPZ, metadata JSON, time-series plot, optional debug
tables, and an annotated MP4.

## Production Path

Keep these pieces together for a clean main branch:

- `scripts/strict_ultratimtrack_gui.py`
- `scripts/run_strict_ultratimtrack_video.py`
- `ultrasound_tracker/`
- `requirements`
- `data/matlab/UTT_numeric_export.mat`
- representative `data/raw/` videos and matching `data/rois/` JSON files
- `tests/`

The GUI defaults to the current best full-run configuration: conditional Hough
localmax fallback, automatic fascicle angle orientation, candidate persistence,
aponeurosis gating, and adaptive anisotropic Kalman with normal Kalman saved as
a comparison.

If you type fascicle angles manually, the GUI uses your fields and disables the
auto-orientation choice for that run..
se
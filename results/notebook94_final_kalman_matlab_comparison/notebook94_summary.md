# Notebook 94 - final MATLAB vs Python Kalman comparison

This run compares MATLAB final outputs against Python normal fixed-R Kalman and Python adaptive anisotropic Kalman.

## Configuration

- Video: `/Users/grosbedou/PycharmProjects/NDORMS/data/raw/UltraTimTrack_test.mp4`
- MATLAB result: `/Users/grosbedou/PycharmProjects/NDORMS/data/matlab/slow_low_01_DOWN_tracked_Q=001.mat`
- Strict result NPZ: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook94_final_kalman_matlab_comparison/python_localmax_adaptive/UltraTimTrack_test/UltraTimTrack_test_strict_results.npz`
- Hough localmax fallback: enabled, mass_below_10deg >= 0.25, gap_to_lower_deg >= 4.0
- Fascicle angle range: 5 to 60 deg
- Compared frames: 2666 with Python offset 0
- Alignment basis: explicit physical-frame alignment; MATLAB timestamps are one-based
- Localmax fallback frames in raw run: 175

## Key metrics

- Best ANG_deg: `Python normal Kalman` RMSE 0.8218 deg
- Best PEN_deg: `Python adaptive Kalman` RMSE 0.8418 deg
- Best FL_mm: `Python adaptive Kalman` RMSE 1.4216 mm

## Outputs

- Metrics CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook94_final_kalman_matlab_comparison/kalman_matlab_metrics.csv`
- Per-frame CSV: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook94_final_kalman_matlab_comparison/kalman_matlab_per_frame.csv`
- Over-time plot: `/Users/grosbedou/PycharmProjects/NDORMS/results/notebook94_final_kalman_matlab_comparison/kalman_matlab_over_time.png`

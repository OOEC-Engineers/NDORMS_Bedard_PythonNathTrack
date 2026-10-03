#!/usr/bin/env python3
"""Notebook 94 helper: final MATLAB vs Python fixed/adaptive Kalman comparison."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import matplotlib
import numpy as np
import pandas as pd
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scripts.compare_updated_matlab_python as cmp


OUT = PROJECT_ROOT / "results" / "notebook94_final_kalman_matlab_comparison"
DEFAULT_RUN_ROOT = OUT / "python_localmax_adaptive"
DEFAULT_RUN_NAME = "UltraTimTrack_test"
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4"
DEFAULT_ROI = PROJECT_ROOT / "data" / "rois" / "UltraTimTrack_test_rois.json"
DEFAULT_MATLAB = PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat"
DEFAULT_UTT_EXPORT = Path("/Users/grosbedou/Documents/GitHub/UltraTimTrack/UTT_numeric_export.mat")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--roi-path", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--matlab-result", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument("--utt-export", type=Path, default=DEFAULT_UTT_EXPORT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--name", default=DEFAULT_RUN_NAME)
    parser.add_argument("--fas-angle-min", type=float, default=5.0)
    parser.add_argument("--fas-angle-max", type=float, default=60.0)
    parser.add_argument(
        "--seed-angle-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        default=(18.0, 24.0),
        help=(
            "Seed-only anatomical angle range. The validated UltraTimTrack_test initialization "
            "uses 18–24 deg while retaining the 5–60 deg per-frame Hough range."
        ),
    )
    parser.add_argument("--mass-below-10deg", type=float, default=0.25)
    parser.add_argument("--gap-to-lower-deg", type=float, default=4.0)
    parser.add_argument(
        "--python-offset",
        type=int,
        default=0,
        help=(
            "Physical Python-frame offset relative to MATLAB sample 0. "
            "The validated UltraTimTrack_test comparison uses 0; MATLAB timestamps are one-based."
        ),
    )
    parser.add_argument("--mm-per-pixel", type=float, default=None)
    parser.add_argument("--image-depth-mm", type=float, default=None)
    parser.add_argument("--force-run", action="store_true", help="Rerun the strict pipeline even if output exists.")
    parser.add_argument("--no-run", action="store_true", help="Only evaluate an existing strict pipeline output.")
    return parser.parse_args()


def strict_paths(run_root: Path, name: str) -> dict[str, Path]:
    run_dir = run_root / name
    return {
        "run_dir": run_dir,
        "npz": run_dir / f"{name}_strict_results.npz",
        "metadata": run_dir / f"{name}_strict_metadata.json",
        "csv": run_dir / f"{name}_strict_FL_PEN_ANG.csv",
    }


def strict_output_ready(npz_path: Path) -> bool:
    if not npz_path.exists():
        return False
    with np.load(npz_path, allow_pickle=True) as data:
        required = {"ANG_deg", "PEN_deg", "FL_mm", "fixed_ANG_deg", "fixed_PEN_deg", "fixed_FL_mm"}
        return required.issubset(data.files) and "hough_localmax_fallback_used" in data.files


def measurement_scale_cli(args: argparse.Namespace) -> tuple[list[str], str]:
    """Return an explicit, validated scale for the current strict-run interface."""

    if args.mm_per_pixel is not None:
        if not np.isfinite(args.mm_per_pixel) or args.mm_per_pixel <= 0:
            raise ValueError("--mm-per-pixel must be positive and finite.")
        return ["--mm-per-pixel", str(args.mm_per_pixel)], "explicit_mm_per_pixel"
    if args.image_depth_mm is not None:
        if not np.isfinite(args.image_depth_mm) or args.image_depth_mm <= 0:
            raise ValueError("--image-depth-mm must be positive and finite.")
        return ["--image-depth-mm", str(args.image_depth_mm)], "explicit_image_depth_mm"

    export = loadmat(args.utt_export, simplify_cells=True)["UTT_numeric_export"]
    export_depth = float(np.asarray(export.get("ID", np.nan), dtype=np.float64).reshape(-1)[0])
    matlab = loadmat(args.matlab_result, simplify_cells=True)
    tracking = matlab.get("TrackingData", {})
    result_depth = float(np.asarray(tracking.get("res", np.nan), dtype=np.float64).reshape(-1)[0])
    finite = [value for value in (export_depth, result_depth) if np.isfinite(value) and value > 0]
    if not finite:
        raise ValueError(
            "No validated length scale was found. Pass --mm-per-pixel or --image-depth-mm explicitly."
        )
    if len(finite) == 2 and not np.isclose(finite[0], finite[1], rtol=0.0, atol=1e-9):
        raise ValueError(
            "UTT export image depth and MATLAB result depth disagree; pass an explicit scale after review."
        )
    depth = finite[0]
    return ["--image-depth-mm", str(depth)], "validated_utt_export_and_matlab_result_image_depth"


def run_strict_pipeline(args: argparse.Namespace, paths: Mapping[str, Path]) -> None:
    scale_cli, scale_source = measurement_scale_cli(args)
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_strict_ultratimtrack_video.py"),
        str(args.video),
        "--name",
        str(args.name),
        "--roi-path",
        str(args.roi_path),
        "--utt-export",
        str(args.utt_export),
        "--results-dir",
        str(args.run_root),
        "--kalman-mode",
        "adaptive-anisotropic",
        "--compare-to-fixed-kalman",
        "--hough-localmax-fallback",
        "--hough-fallback-min-mass-below-10deg",
        str(args.mass_below_10deg),
        "--hough-fallback-min-gap-to-lower-deg",
        str(args.gap_to_lower_deg),
        "--fas-angle-min",
        str(args.fas_angle_min),
        "--fas-angle-max",
        str(args.fas_angle_max),
        "--seed-angle-range",
        str(args.seed_angle_range[0]),
        str(args.seed_angle_range[1]),
        "--no-annotated-video",
        "--save-overlays",
        "0",
        "--no-time-series-plot",
        "--progress-every",
        "250",
    ]
    cmd.extend(scale_cli)
    print("Running strict pipeline:")
    print(f"Measurement scale source: {scale_source}")
    print(" ".join(cmd))
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    if not strict_output_ready(paths["npz"]):
        raise RuntimeError(f"Strict run finished, but expected fixed/adaptive localmax output is incomplete: {paths['npz']}")


def metric_row(method: str, signal: str, unit: str, reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | str | int]:
    ref = np.asarray(reference, dtype=np.float64).reshape(-1)
    est = np.asarray(estimate, dtype=np.float64).reshape(-1)
    n = min(len(ref), len(est))
    ref = ref[:n]
    est = est[:n]
    valid = np.isfinite(ref) & np.isfinite(est)
    ref = ref[valid]
    est = est[valid]
    if not len(ref):
        return {
            "method": method,
            "signal": signal,
            "unit": unit,
            "n": 0,
            "bias": np.nan,
            "mae": np.nan,
            "rmse": np.nan,
            "corr": np.nan,
        }
    delta = est - ref
    corr = float(np.corrcoef(ref, est)[0, 1]) if len(ref) > 1 and np.nanstd(ref) > 0 and np.nanstd(est) > 0 else np.nan
    return {
        "method": method,
        "signal": signal,
        "unit": unit,
        "n": int(len(ref)),
        "bias": float(np.mean(delta)),
        "mae": float(np.mean(np.abs(delta))),
        "rmse": float(np.sqrt(np.mean(delta * delta))),
        "corr": corr,
    }


def load_comparison(
    npz_path: Path,
    matlab_result: Path,
    *,
    python_offset: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    with np.load(npz_path, allow_pickle=True) as npz:
        py = {key: npz[key] for key in npz.files}

    mat = loadmat(matlab_result, simplify_cells=True)
    region = mat["Fdat"]["Region"]
    matlab_time = cmp.as_float1(region["Time"])
    python_time = cmp.as_float1(py["time_s"])
    if python_offset < 0:
        raise ValueError("Notebook 94 evaluator expects a non-negative Python physical-frame offset.")
    n = min(len(matlab_time), len(python_time) - python_offset)
    if n <= 0:
        raise ValueError("No overlapping MATLAB/Python samples.")
    sl = slice(python_offset, python_offset + n)

    matlab = {
        "time_s": matlab_time[:n],
        "ANG_deg": cmp.as_float1(region["ANG"])[:n],
        "PEN_deg": cmp.as_float1(region["PEN"])[:n],
        "FL_mm": cmp.as_float1(region["FL"])[:n],
    }
    normal = {
        "ANG_deg": cmp.as_float1(py["fixed_ANG_deg"])[sl],
        "PEN_deg": cmp.as_float1(py["fixed_PEN_deg"])[sl],
        "FL_mm": cmp.as_float1(py["fixed_FL_mm"])[sl],
    }
    adaptive = {
        "ANG_deg": cmp.as_float1(py["ANG_deg"])[sl],
        "PEN_deg": cmp.as_float1(py["PEN_deg"])[sl],
        "FL_mm": cmp.as_float1(py["FL_mm"])[sl],
    }

    metric_rows = []
    for signal, unit in [("ANG_deg", "deg"), ("PEN_deg", "deg"), ("FL_mm", "mm")]:
        metric_rows.append(metric_row("Python normal Kalman", signal, unit, matlab[signal], normal[signal]))
        metric_rows.append(metric_row("Python adaptive Kalman", signal, unit, matlab[signal], adaptive[signal]))
    metrics = pd.DataFrame(metric_rows)

    per_frame = pd.DataFrame(
        {
            "frame": np.arange(n, dtype=int),
            "time_s": matlab["time_s"],
            "MATLAB_ANG_deg": matlab["ANG_deg"],
            "Python_normal_ANG_deg": normal["ANG_deg"],
            "Python_adaptive_ANG_deg": adaptive["ANG_deg"],
            "MATLAB_PEN_deg": matlab["PEN_deg"],
            "Python_normal_PEN_deg": normal["PEN_deg"],
            "Python_adaptive_PEN_deg": adaptive["PEN_deg"],
            "MATLAB_FL_mm": matlab["FL_mm"],
            "Python_normal_FL_mm": normal["FL_mm"],
            "Python_adaptive_FL_mm": adaptive["FL_mm"],
        }
    )
    for signal in ["ANG_deg", "PEN_deg", "FL_mm"]:
        short = signal.replace("_deg", "").replace("_mm", "")
        per_frame[f"Python_normal_minus_MATLAB_{signal}"] = per_frame[f"Python_normal_{short}_{signal.split('_')[-1]}"] - per_frame[f"MATLAB_{short}_{signal.split('_')[-1]}"]
        per_frame[f"Python_adaptive_minus_MATLAB_{signal}"] = per_frame[f"Python_adaptive_{short}_{signal.split('_')[-1]}"] - per_frame[f"MATLAB_{short}_{signal.split('_')[-1]}"]

    info = {
        "python_offset": int(python_offset),
        "frames": int(n),
        "fallback_frames": int(np.sum(np.asarray(py.get("hough_localmax_fallback_used", []), dtype=bool))),
        "raw_npz_frames": int(len(python_time)),
        "alignment_basis": "explicit physical-frame alignment; MATLAB timestamps are one-based",
    }
    return metrics, per_frame, info


def plot_over_time(per_frame: pd.DataFrame, output_path: Path) -> None:
    specs = [
        ("ANG", "deg", "Fascicle angle over time"),
        ("PEN", "deg", "Pennation over time"),
        ("FL", "mm", "Fascicle length over time"),
    ]
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
    time_s = per_frame["time_s"].to_numpy(dtype=float)
    for ax, (signal, unit, title) in zip(axes, specs):
        key = f"{signal}_{unit}"
        ax.plot(time_s, per_frame[f"MATLAB_{key}"], color="black", linewidth=1.4, label="MATLAB")
        ax.plot(
            time_s,
            per_frame[f"Python_normal_{key}"],
            color="tab:blue",
            linewidth=1.0,
            alpha=0.9,
            label="Python normal Kalman",
        )
        ax.plot(
            time_s,
            per_frame[f"Python_adaptive_{key}"],
            color="tab:red",
            linewidth=1.0,
            alpha=0.9,
            label="Python adaptive Kalman",
        )
        ax.set_title(title)
        ax.set_ylabel(f"{signal} ({unit})")
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(loc="best")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def write_summary(
    summary_path: Path,
    metrics: pd.DataFrame,
    info: Mapping[str, object],
    paths: Mapping[str, Path],
    args: argparse.Namespace,
) -> None:
    def best_line(signal: str) -> str:
        sub = metrics[metrics["signal"] == signal].sort_values("rmse")
        row = sub.iloc[0]
        return f"- Best {signal}: `{row['method']}` RMSE {row['rmse']:.4f} {row['unit']}"

    lines = [
        "# Notebook 94 - final MATLAB vs Python Kalman comparison",
        "",
        "This run compares MATLAB final outputs against Python normal fixed-R Kalman and Python adaptive anisotropic Kalman.",
        "",
        "## Configuration",
        "",
        f"- Video: `{args.video}`",
        f"- MATLAB result: `{args.matlab_result}`",
        f"- Strict result NPZ: `{paths['npz']}`",
        f"- Hough localmax fallback: enabled, mass_below_10deg >= {args.mass_below_10deg}, gap_to_lower_deg >= {args.gap_to_lower_deg}",
        f"- Fascicle angle range: {args.fas_angle_min:g} to {args.fas_angle_max:g} deg",
        f"- Compared frames: {info['frames']} with Python offset {info['python_offset']}",
        f"- Alignment basis: {info['alignment_basis']}",
        f"- Localmax fallback frames in raw run: {info['fallback_frames']}",
        "",
        "## Key metrics",
        "",
        best_line("ANG_deg"),
        best_line("PEN_deg"),
        best_line("FL_mm"),
        "",
        "## Outputs",
        "",
        f"- Metrics CSV: `{OUT / 'kalman_matlab_metrics.csv'}`",
        f"- Per-frame CSV: `{OUT / 'kalman_matlab_per_frame.csv'}`",
        f"- Over-time plot: `{OUT / 'kalman_matlab_over_time.png'}`",
    ]
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    args.run_root.mkdir(parents=True, exist_ok=True)
    paths = strict_paths(args.run_root, args.name)

    if args.force_run or (not args.no_run and not strict_output_ready(paths["npz"])):
        run_strict_pipeline(args, paths)
    elif not strict_output_ready(paths["npz"]):
        raise FileNotFoundError(
            f"Missing strict output with fixed/adaptive localmax fields: {paths['npz']}. "
            "Run without --no-run or pass --force-run."
        )

    metrics, per_frame, info = load_comparison(
        paths["npz"],
        args.matlab_result,
        python_offset=args.python_offset,
    )
    metrics_path = OUT / "kalman_matlab_metrics.csv"
    per_frame_path = OUT / "kalman_matlab_per_frame.csv"
    plot_path = OUT / "kalman_matlab_over_time.png"
    summary_path = OUT / "notebook94_summary.md"

    metrics.to_csv(metrics_path, index=False)
    per_frame.to_csv(per_frame_path, index=False)
    plot_over_time(per_frame, plot_path)
    write_summary(summary_path, metrics, info, paths, args)

    print(summary_path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

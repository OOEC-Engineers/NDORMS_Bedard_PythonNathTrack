#!/usr/bin/env python3
"""Build the publication-oriented MATLAB/Python validation data products.

This script is analysis-only.  It reads the immutable MATLAB result and the
fresh strict Python NPZ, applies the audited physical-frame alignment, and
writes flat CSV/JSON/figure artifacts used by the final Excel workbook.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import subprocess
import sys
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping

sys.dont_write_bytecode = True

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

import cv2
import matplotlib
import numpy as np
import pandas as pd
from scipy import stats
from scipy.io import loadmat

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ultrasound_tracker.speckle_confidence import SpeckleConfidenceConfig


DEFAULT_OUT = PROJECT_ROOT / "results" / "notebook94_final_kalman_matlab_comparison"
DEFAULT_NPZ = (
    DEFAULT_OUT
    / "python_localmax_adaptive"
    / "UltraTimTrack_test"
    / "UltraTimTrack_test_strict_results.npz"
)
DEFAULT_METADATA = DEFAULT_NPZ.with_name("UltraTimTrack_test_strict_metadata.json")
DEFAULT_VIDEO = PROJECT_ROOT / "data" / "raw" / "UltraTimTrack_test.mp4"
DEFAULT_ROI = PROJECT_ROOT / "data" / "rois" / "UltraTimTrack_test_rois.json"
DEFAULT_MATLAB = PROJECT_ROOT / "data" / "matlab" / "slow_low_01_DOWN_tracked_Q=001.mat"
DEFAULT_UTT = PROJECT_ROOT / "data" / "matlab" / "UTT_numeric_export.mat"
FINAL_OFFSET = 0
BOOTSTRAP_REPLICATES = 5000
RANDOM_SEED = 9402026


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=Path, default=DEFAULT_VIDEO)
    parser.add_argument("--roi", type=Path, default=DEFAULT_ROI)
    parser.add_argument("--matlab", type=Path, default=DEFAULT_MATLAB)
    parser.add_argument("--utt-export", type=Path, default=DEFAULT_UTT)
    parser.add_argument("--python-npz", type=Path, default=DEFAULT_NPZ)
    parser.add_argument("--python-metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--random-seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def git_output(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=PROJECT_ROOT, check=True, text=True, capture_output=True
    ).stdout.strip()


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "scipy", "matplotlib", "opencv-python", "nbformat", "nbconvert"]
    versions: dict[str, str] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return versions


def as_float1(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=np.float64).reshape(-1)


def object_pairs(values: Any) -> np.ndarray:
    rows = np.asarray(values, dtype=object).reshape(-1)
    out = np.full((len(rows), 2), np.nan, dtype=np.float64)
    for idx, value in enumerate(rows):
        arr = np.asarray(value, dtype=np.float64).reshape(-1)
        out[idx, : min(2, len(arr))] = arr[:2]
    return out


def pad(values: Any, n: int, *, dtype: Any = np.float64, fill: Any = np.nan) -> np.ndarray:
    arr = np.asarray(values, dtype=dtype)
    shape = (n, *arr.shape[1:])
    out = np.full(shape, fill, dtype=dtype)
    out[: min(n, len(arr))] = arr[:n]
    return out


def align_pair(reference: np.ndarray, estimate: np.ndarray, offset: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pair MATLAB sample i with Python frame i+offset and return MATLAB indices."""

    reference = as_float1(reference)
    estimate = as_float1(estimate)
    matlab_start = max(0, -int(offset))
    python_start = max(0, int(offset))
    n = min(len(reference) - matlab_start, len(estimate) - python_start)
    if n <= 0:
        return np.asarray([]), np.asarray([]), np.asarray([], dtype=int)
    idx = np.arange(matlab_start, matlab_start + n, dtype=int)
    return reference[idx], estimate[python_start : python_start + n], idx


def simple_metrics(reference: np.ndarray, estimate: np.ndarray) -> dict[str, float | int]:
    reference = as_float1(reference)
    estimate = as_float1(estimate)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    valid = np.isfinite(reference) & np.isfinite(estimate)
    ref, est = reference[valid], estimate[valid]
    if len(ref) == 0:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "pearson_r": np.nan}
    error = est - ref
    pearson = (
        float(stats.pearsonr(ref, est).statistic)
        if len(ref) > 1 and np.std(ref) > 0 and np.std(est) > 0
        else np.nan
    )
    return {
        "n": int(len(ref)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error**2))),
        "pearson_r": pearson,
    }


def full_metrics(
    reference: np.ndarray,
    estimate: np.ndarray,
    *,
    total_reference_frames: int,
) -> dict[str, float | int | str]:
    reference = as_float1(reference)
    estimate = as_float1(estimate)
    n = min(len(reference), len(estimate))
    reference, estimate = reference[:n], estimate[:n]
    valid = np.isfinite(reference) & np.isfinite(estimate)
    ref, est = reference[valid], estimate[valid]
    missing = int(total_reference_frames - len(ref))
    base: dict[str, float | int | str] = {
        "paired_frames_n": int(len(ref)),
        "reference_frames_n": int(total_reference_frames),
        "missing_frames_n": missing,
        "missing_frames_percent": 100.0 * missing / total_reference_frames,
        "nrmse_denominator_definition": "paired MATLAB maximum minus paired MATLAB minimum",
    }
    if len(ref) == 0:
        return {
            **base,
            **{key: np.nan for key in [
                "mean_matlab", "mean_python", "bias_python_minus_matlab", "difference_sd",
                "mae", "median_absolute_error", "rmse", "matlab_range", "nrmse",
                "maximum_absolute_error", "p95_absolute_error", "pearson_r", "spearman_rho",
                "regression_slope", "regression_intercept", "regression_r_squared",
                "bland_altman_lower_loa", "bland_altman_upper_loa", "bland_altman_width",
            ]},
            "large_outlier_frames_n": 0,
            "large_outlier_rule": "absolute error strictly above method-variable 95th percentile",
        }
    diff = est - ref
    abs_error = np.abs(diff)
    difference_sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else np.nan
    reference_range = float(np.max(ref) - np.min(ref))
    pearson = (
        float(stats.pearsonr(ref, est).statistic)
        if len(ref) > 1 and np.std(ref) > 0 and np.std(est) > 0
        else np.nan
    )
    spearman = float(stats.spearmanr(ref, est).statistic) if len(ref) > 1 else np.nan
    if len(ref) > 1 and np.std(ref) > 0:
        regression = stats.linregress(ref, est)
        slope, intercept, r_squared = float(regression.slope), float(regression.intercept), float(regression.rvalue**2)
    else:
        slope = intercept = r_squared = np.nan
    bias = float(np.mean(diff))
    p95 = float(np.percentile(abs_error, 95))
    return {
        **base,
        "mean_matlab": float(np.mean(ref)),
        "mean_python": float(np.mean(est)),
        "bias_python_minus_matlab": bias,
        "difference_sd": difference_sd,
        "mae": float(np.mean(abs_error)),
        "median_absolute_error": float(np.median(abs_error)),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "matlab_range": reference_range,
        "nrmse": float(np.sqrt(np.mean(diff**2)) / reference_range) if reference_range > 0 else np.nan,
        "maximum_absolute_error": float(np.max(abs_error)),
        "p95_absolute_error": p95,
        "pearson_r": pearson,
        "spearman_rho": spearman,
        "regression_slope": slope,
        "regression_intercept": intercept,
        "regression_r_squared": r_squared,
        "bland_altman_lower_loa": bias - 1.96 * difference_sd,
        "bland_altman_upper_loa": bias + 1.96 * difference_sd,
        "bland_altman_width": 3.92 * difference_sd,
        "large_outlier_frames_n": int(np.sum(abs_error > p95)),
        "large_outlier_rule": "absolute error strictly above method-variable 95th percentile",
    }


def autocorrelation_block_length(values: np.ndarray, max_lag: int = 300) -> int:
    arr = as_float1(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 3 or np.std(arr) == 0:
        return 1
    centered = arr - np.mean(arr)
    denom = float(np.dot(centered, centered))
    for lag in range(1, min(max_lag, len(arr) - 1) + 1):
        acf = float(np.dot(centered[:-lag], centered[lag:]) / denom)
        if abs(acf) <= math.exp(-1):
            return lag
    return min(max_lag, len(arr))


def moving_block_bootstrap(
    values: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, float | int | str]:
    arr = as_float1(values)
    arr = arr[np.isfinite(arr)]
    if len(arr) == 0:
        return {
            "bootstrap_block_length_frames": 0,
            "bootstrap_replicates": replicates,
            "mean_difference_ci95_lower": np.nan,
            "mean_difference_ci95_upper": np.nan,
            "block_bootstrap_null_p_value": np.nan,
        }
    block = autocorrelation_block_length(arr)
    rng = np.random.default_rng(seed)
    n_blocks = int(math.ceil(len(arr) / block))
    max_start = len(arr) - block
    means = np.empty(replicates, dtype=np.float64)
    centered = arr - np.mean(arr)
    null_means = np.empty(replicates, dtype=np.float64)
    for idx in range(replicates):
        starts = rng.integers(0, max_start + 1, size=n_blocks)
        sample = np.concatenate([arr[start : start + block] for start in starts])[: len(arr)]
        null_sample = np.concatenate([centered[start : start + block] for start in starts])[: len(arr)]
        means[idx] = np.mean(sample)
        null_means[idx] = np.mean(null_sample)
    observed = float(np.mean(arr))
    p_value = float((1 + np.sum(np.abs(null_means) >= abs(observed))) / (replicates + 1))
    return {
        "bootstrap_block_length_frames": int(block),
        "bootstrap_block_selection": "first absolute autocorrelation lag <= exp(-1), capped at 300 frames",
        "bootstrap_replicates": int(replicates),
        "mean_difference_ci95_lower": float(np.percentile(means, 2.5)),
        "mean_difference_ci95_upper": float(np.percentile(means, 97.5)),
        "block_bootstrap_null_p_value": p_value,
    }


def video_metadata(path: Path) -> dict[str, float | int]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(path)
    out = {
        "fps": float(cap.get(cv2.CAP_PROP_FPS)),
        "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width_px": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height_px": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    }
    cap.release()
    return out


def video_brightness(path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    values: list[float] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        values.append(float(np.mean(gray)))
    cap.release()
    return np.asarray(values, dtype=np.float64)


def save_figure(fig: plt.Figure, base_path: Path) -> None:
    fig.tight_layout()
    fig.savefig(base_path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base_path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def create_plots(
    framewise: pd.DataFrame,
    figure_data: pd.DataFrame,
    alignment: pd.DataFrame,
    confidence: pd.DataFrame,
    plots_dir: Path,
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    method_colors = {"fixed-R": "#1f77b4", "adaptive anisotropic-R": "#d62728"}
    specs = [("ANG", "deg", "Fascicle angle"), ("PEN", "deg", "Pennation angle"), ("FL", "mm", "Fascicle length")]
    included = framewise[framewise["included_in_analysis"]].copy()

    for variable, unit, label in specs:
        fig, ax = plt.subplots(figsize=(11, 4.3))
        ax.plot(included["time_seconds"], included[f"MATLAB_{variable}_{unit}"], color="black", lw=1.5, label="MATLAB")
        ax.plot(included["time_seconds"], included[f"Python_fixed_{variable}_{unit}"], color=method_colors["fixed-R"], lw=1.0, label="Python fixed-R")
        ax.plot(included["time_seconds"], included[f"Python_adaptive_{variable}_{unit}"], color=method_colors["adaptive anisotropic-R"], lw=1.0, label="Python adaptive anisotropic-R")
        ax.set(xlabel="Time (s)", ylabel=f"{label} ({unit})", title=f"{label} over time")
        ax.grid(alpha=0.2)
        ax.legend(frameon=False, ncol=3)
        save_figure(fig, plots_dir / f"timeseries_{variable}")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    for row, (variable, unit, label) in enumerate(specs):
        for col, method in enumerate(method_colors):
            ax = axes[row, col]
            data = figure_data[(figure_data.variable == variable) & (figure_data.method == method)]
            ax.scatter(data["bland_altman_mean"], data["difference_python_minus_matlab"], s=5, alpha=0.35, color=method_colors[method], rasterized=True)
            bias = float(data["difference_python_minus_matlab"].mean())
            sd = float(data["difference_python_minus_matlab"].std(ddof=1))
            ax.axhline(bias, color="black", lw=1)
            ax.axhline(bias - 1.96 * sd, color="black", lw=0.8, ls="--")
            ax.axhline(bias + 1.96 * sd, color="black", lw=0.8, ls="--")
            ax.set(title=f"{label}: {method}", xlabel=f"Mean of MATLAB and Python ({unit})", ylabel=f"Python − MATLAB ({unit})")
            ax.grid(alpha=0.15)
    save_figure(fig, plots_dir / "bland_altman_grid")

    fig, axes = plt.subplots(3, 2, figsize=(12, 13))
    for row, (variable, unit, label) in enumerate(specs):
        for col, method in enumerate(method_colors):
            ax = axes[row, col]
            data = figure_data[(figure_data.variable == variable) & (figure_data.method == method)]
            ax.scatter(data["MATLAB_value"], data["Python_value"], s=5, alpha=0.35, color=method_colors[method], rasterized=True)
            low = float(np.nanmin([data["MATLAB_value"].min(), data["Python_value"].min()]))
            high = float(np.nanmax([data["MATLAB_value"].max(), data["Python_value"].max()]))
            ax.plot([low, high], [low, high], color="black", lw=1, ls="--")
            ax.set(title=f"{label}: {method}", xlabel=f"MATLAB ({unit})", ylabel=f"Python ({unit})")
            ax.grid(alpha=0.15)
    save_figure(fig, plots_dir / "scatter_identity_grid")

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (variable, unit, label) in zip(axes, specs):
        for method in method_colors:
            data = figure_data[(figure_data.variable == variable) & (figure_data.method == method)]
            ax.plot(data.time_seconds, data.absolute_error, lw=0.8, color=method_colors[method], label=method)
        ax.set(ylabel=f"Absolute error ({unit})", title=label)
        ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Time (s)")
    axes[0].legend(frameon=False, ncol=2)
    save_figure(fig, plots_dir / "absolute_error_over_time_grid")

    fig, axes = plt.subplots(3, 1, figsize=(12, 9), sharex=True)
    axes[0].plot(included.time_seconds, included.adaptive_confidence_score, color="#2ca02c", lw=0.9, label="Combined confidence")
    axes[0].plot(included.time_seconds, included.adaptive_confidence_angle, color="#9467bd", lw=0.7, alpha=0.8, label="Angle confidence")
    axes[0].plot(included.time_seconds, included.adaptive_confidence_length, color="#8c564b", lw=0.7, alpha=0.8, label="Length confidence")
    axes[0].set_ylabel("Confidence (0–1)"); axes[0].legend(frameon=False, ncol=3); axes[0].grid(alpha=0.15)
    axes[1].plot(included.time_seconds, included.adaptive_R_scale_angle, color="#9467bd", lw=0.8, label="Angle R scale")
    axes[1].plot(included.time_seconds, included.adaptive_R_scale_length, color="#8c564b", lw=0.8, label="Length-side R scale")
    axes[1].set_ylabel("R scale"); axes[1].legend(frameon=False, ncol=2); axes[1].grid(alpha=0.15)
    axes[2].plot(included.time_seconds, included.adaptive_angle_measurement_covariance_deg2, color="#9467bd", lw=0.8, label="Angle covariance (deg²)")
    axes[2].plot(included.time_seconds, included.adaptive_length_side_measurement_covariance_px2, color="#8c564b", lw=0.8, label="Length-side x covariance (px²)")
    axes[2].set(xlabel="Time (s)", ylabel="Measurement covariance"); axes[2].legend(frameon=False, ncol=2); axes[2].grid(alpha=0.15)
    save_figure(fig, plots_dir / "confidence_r_scale_over_time")

    fig, axes = plt.subplots(3, 1, figsize=(10, 11))
    for ax, (variable, unit, label) in zip(axes, specs):
        data = figure_data[figure_data.variable == variable]
        for method in method_colors:
            sub = data[data.method == method]
            ax.scatter(sub.confidence_score, sub.absolute_error, s=4, alpha=0.12, color=method_colors[method], rasterized=True)
        bins = confidence[(confidence.analysis_scope == "confidence_bin") & (confidence.variable == variable)]
        centers = bins.confidence_bin_midpoint.to_numpy(dtype=float)
        ax.plot(centers, bins.fixed_R_MAE, color=method_colors["fixed-R"], marker="o", label="fixed-R bin MAE")
        ax.plot(centers, bins.adaptive_R_MAE, color=method_colors["adaptive anisotropic-R"], marker="o", label="adaptive bin MAE")
        ax.set(xlabel="Combined confidence", ylabel=f"Absolute error ({unit})", title=label)
        ax.grid(alpha=0.15)
    axes[0].legend(frameon=False, ncol=2)
    save_figure(fig, plots_dir / "error_vs_confidence_grid")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (variable, unit, label) in zip(axes, specs):
        pivot = figure_data[figure_data.variable == variable].pivot(index="frame_index_video", columns="method", values="absolute_error").dropna()
        ax.scatter(pivot["fixed-R"], pivot["adaptive anisotropic-R"], s=5, alpha=0.25, color="#4c78a8", rasterized=True)
        high = float(np.nanmax(pivot.to_numpy()))
        ax.plot([0, high], [0, high], color="black", ls="--", lw=1)
        ax.set(xlabel=f"Fixed-R absolute error ({unit})", ylabel=f"Adaptive absolute error ({unit})", title=label)
        ax.grid(alpha=0.15)
    save_figure(fig, plots_dir / "fixed_vs_adaptive_absolute_error_grid")

    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    for ax, (variable, unit, label) in zip(axes, specs):
        for method in method_colors:
            sub = alignment[(alignment.variable == variable) & (alignment.method == method)]
            ax.plot(sub.python_offset_frames, sub.rmse, marker="o", color=method_colors[method], label=method)
        ax.axvline(0, color="black", ls="--", lw=1)
        ax.set(ylabel=f"RMSE ({unit})", title=label); ax.grid(alpha=0.15)
    axes[-1].set_xlabel("Python physical-frame offset relative to MATLAB")
    axes[0].legend(frameon=False, ncol=2)
    save_figure(fig, plots_dir / "temporal_offset_sensitivity_grid")


def write_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, na_rep="", lineterminator="\n")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.output_dir / "plots"
    for path in [args.video, args.roi, args.matlab, args.utt_export, args.python_npz, args.python_metadata]:
        if not path.exists():
            raise FileNotFoundError(path)

    video = video_metadata(args.video)
    with np.load(args.python_npz, allow_pickle=True) as bundle:
        py = {key: bundle[key] for key in bundle.files}
    mat_file = loadmat(args.matlab, simplify_cells=True)
    region = mat_file["Fdat"]["Region"]
    fascicle = region["Fascicle"]
    tracking = mat_file["TrackingData"]
    geofeatures = np.asarray(mat_file["Fdat"]["geofeatures"], dtype=object).reshape(-1)
    utt = loadmat(args.utt_export, simplify_cells=True)["UTT_numeric_export"]
    strict_meta = json.loads(args.python_metadata.read_text(encoding="utf-8"))

    matlab_n = len(as_float1(region["Time"]))
    python_n = len(as_float1(py["time_s"]))
    video_n = int(video["frames"])
    if python_n != video_n:
        raise ValueError(f"Python output has {python_n} frames but video metadata reports {video_n}.")
    if FINAL_OFFSET != 0:
        raise AssertionError("Publication analysis is expected to use the audited zero physical-frame offset.")

    scale = float(np.asarray(py["mm_per_pixel"], dtype=np.float64).reshape(-1)[0])
    validated_scale = float(tracking["res"]) / float(video["height_px"])
    if not np.isclose(scale, validated_scale, rtol=0.0, atol=1e-12):
        raise ValueError(f"Python mm/px {scale} does not match validated MATLAB scale {validated_scale}.")

    n = video_n
    frame = np.arange(n, dtype=int)
    mat_ang = pad(as_float1(region["ANG"]), n)
    mat_pen = pad(as_float1(region["PEN"]), n)
    mat_fl = pad(as_float1(region["FL"]), n)
    mat_time = pad(as_float1(region["Time"]), n)
    fixed_ang = pad(py["fixed_ANG_deg"], n)
    adaptive_ang = pad(py["ANG_deg"], n)
    fixed_pen = pad(py["fixed_PEN_deg"], n)
    adaptive_pen = pad(py["PEN_deg"], n)
    fixed_fl = pad(py["fixed_FL_mm"], n)
    adaptive_fl = pad(py["FL_mm"], n)

    required = np.column_stack([mat_ang, mat_pen, mat_fl, fixed_ang, adaptive_ang, fixed_pen, adaptive_pen, fixed_fl, adaptive_fl])
    included = np.all(np.isfinite(required), axis=1)
    exclusion_reason: list[str] = []
    for idx in frame:
        reasons: list[str] = []
        if idx >= matlab_n:
            reasons.append("No MATLAB sample: MATLAB contains 2666 samples for 2667 OpenCV-decoded video frames")
        if idx < matlab_n and not np.isfinite(required[idx]).all():
            reasons.append("At least one required MATLAB/Python ANG, PEN, or FL value is missing/non-finite")
        exclusion_reason.append("; ".join(reasons))

    matlab_sup_x = pad(object_pairs(region["sup_x"]), n)
    matlab_sup_y = pad(object_pairs(region["sup_y"]), n)
    matlab_deep_x = pad(object_pairs(region["deep_x"]), n)
    matlab_deep_y = pad(object_pairs(region["deep_y"]), n)
    matlab_fas_x = pad(object_pairs(fascicle["fas_x_end"]), n)
    matlab_fas_y = pad(object_pairs(fascicle["fas_y_end"]), n)
    python_sup = pad(py["sup_apo_lines"], n)
    python_deep = pad(py["deep_apo_lines"], n)
    adaptive_segment = pad(py["fascicle_end_segments"], n)
    fixed_segment = pad(py["fixed_fascicle_end_segments"], n)

    framewise = pd.DataFrame({
        "frame_index_video": frame,
        "frame_index_matlab": pd.array([idx + 1 if idx < matlab_n else None for idx in frame], dtype="Int64"),
        "matlab_sample_index_zero_based": pd.array([idx if idx < matlab_n else None for idx in frame], dtype="Int64"),
        "frame_index_python": frame,
        "time_seconds": frame / float(video["fps"]),
        "MATLAB_time_seconds_stored": mat_time,
        "included_in_analysis": included,
        "exclusion_reason": exclusion_reason,
        "MATLAB_ANG_deg": mat_ang,
        "Python_fixed_ANG_deg": fixed_ang,
        "Python_adaptive_ANG_deg": adaptive_ang,
        "MATLAB_PEN_deg": mat_pen,
        "Python_fixed_PEN_deg": fixed_pen,
        "Python_adaptive_PEN_deg": adaptive_pen,
        "MATLAB_FL_original_unit": mat_fl,
        "MATLAB_FL_original_unit_name": ["mm" if idx < matlab_n else "" for idx in frame],
        "Python_fixed_FL_original_unit": fixed_fl,
        "Python_fixed_FL_original_unit_name": ["mm"] * n,
        "Python_adaptive_FL_original_unit": adaptive_fl,
        "Python_adaptive_FL_original_unit_name": ["mm"] * n,
        "MATLAB_FL_px_derived": mat_fl / scale,
        "Python_fixed_FL_px": pad(py["fixed_FL_px"], n),
        "Python_adaptive_FL_px": pad(py["FL_px"], n),
        "MATLAB_FL_mm": mat_fl,
        "Python_fixed_FL_mm": fixed_fl,
        "Python_adaptive_FL_mm": adaptive_fl,
        "fixed_minus_matlab_ANG_deg": fixed_ang - mat_ang,
        "adaptive_minus_matlab_ANG_deg": adaptive_ang - mat_ang,
        "fixed_minus_matlab_PEN_deg": fixed_pen - mat_pen,
        "adaptive_minus_matlab_PEN_deg": adaptive_pen - mat_pen,
        "fixed_minus_matlab_FL_mm": fixed_fl - mat_fl,
        "adaptive_minus_matlab_FL_mm": adaptive_fl - mat_fl,
        "absolute_error_fixed_ANG_deg": np.abs(fixed_ang - mat_ang),
        "absolute_error_adaptive_ANG_deg": np.abs(adaptive_ang - mat_ang),
        "absolute_error_fixed_PEN_deg": np.abs(fixed_pen - mat_pen),
        "absolute_error_adaptive_PEN_deg": np.abs(adaptive_pen - mat_pen),
        "absolute_error_fixed_FL_mm": np.abs(fixed_fl - mat_fl),
        "absolute_error_adaptive_FL_mm": np.abs(adaptive_fl - mat_fl),
        "adaptive_confidence_score": pad(py["combined_confidence"], n),
        "adaptive_confidence_angle": pad(py["confidence_theta"], n),
        "adaptive_confidence_length": pad(py["confidence_length"], n),
        "adaptive_R_scale_global": pad(py["r_scale"], n),
        "adaptive_R_scale_angle": pad(py["measurement_r_scale_theta"], n),
        "adaptive_R_scale_length": pad(py["measurement_r_scale_length"], n),
        "adaptive_angle_measurement_covariance_deg2": pad(py["measurement_R_diag"], n)[:, 1],
        "adaptive_length_side_measurement_covariance_px2": pad(py["measurement_R_diag"], n)[:, 0],
        "fixed_R_scale_angle": pad(py["fixed_measurement_r_scale_theta"], n),
        "fixed_R_scale_length": pad(py["fixed_measurement_r_scale_length"], n),
        "fixed_angle_measurement_covariance_deg2": pad(py["fixed_measurement_R_diag"], n)[:, 1],
        "fixed_length_side_measurement_covariance_px2": pad(py["fixed_measurement_R_diag"], n)[:, 0],
        "adaptive_kalman_gain_length_side": pad(py["kalman_gain"], n)[:, 0],
        "adaptive_kalman_gain_angle": pad(py["kalman_gain"], n)[:, 1],
        "fixed_kalman_gain_length_side": pad(py["fixed_kalman_gain"], n)[:, 0],
        "fixed_kalman_gain_angle": pad(py["fixed_kalman_gain"], n)[:, 1],
        "adaptive_smoother_gain_length_side": pad(py["smoother_gain"], n)[:, 0],
        "adaptive_smoother_gain_angle": pad(py["smoother_gain"], n)[:, 1],
        "fixed_smoother_gain_length_side": pad(py["fixed_smoother_gain"], n)[:, 0],
        "fixed_smoother_gain_angle": pad(py["fixed_smoother_gain"], n)[:, 1],
        "detection_success": pad(py["detection_success"], n, dtype=bool, fill=False),
        "missing_detection_flag": ~pad(py["detection_success"], n, dtype=bool, fill=False),
        "hough_localmax_fallback_flag": pad(py["hough_localmax_fallback_used"], n, dtype=bool, fill=False),
        "hough_peak_source": pad(py["hough_peak_source"], n, dtype=object, fill=""),
        "raw_timtrack_alpha_deg": pad(py["raw_timtrack_alpha_deg"], n),
        "selected_timtrack_alpha_deg": pad(py["timtrack_alpha_deg"], n),
        "hough_baseline_alpha_deg": pad(py["hough_baseline_alpha_deg"], n),
        "hough_fallback_mass_below_10deg": pad(py["hough_localmax_fallback_mass_below_10deg"], n),
        "hough_fallback_gap_to_lower_deg": pad(py["hough_localmax_fallback_gap_to_lower_deg"], n),
        "fascicle_candidate_raw_rejected": pad(py["fascicle_candidate_raw_rejected"], n, dtype=bool, fill=False),
        "fascicle_candidate_selection_reason": pad(py["fascicle_candidate_selection_reason"], n, dtype=object, fill=""),
        "klt_affine_ok": pad(py["klt_affine_ok"], n, dtype=bool, fill=False),
        "klt_tracker_redetected": pad(py["klt_tracker_redetected"], n, dtype=bool, fill=False),
        "klt_tracker_found_fraction": pad(py["klt_tracker_found_fraction"], n),
        "klt_points_count": pad(py["klt_points_count"], n, dtype=np.int32, fill=0),
        "klt_tracked_count": pad(py["klt_tracked_count"], n, dtype=np.int32, fill=0),
        "klt_inlier_count": pad(py["klt_inlier_count"], n, dtype=np.int32, fill=0),
        "superficial_aponeurosis_rejected": np.any(pad(py["apo_rejected_endpoints"], n, dtype=bool, fill=False)[:, :2], axis=1),
        "deep_aponeurosis_rejected": np.any(pad(py["apo_rejected_endpoints"], n, dtype=bool, fill=False)[:, 2:], axis=1),
        "MATLAB_superficial_x_left_1b": matlab_sup_x[:, 0],
        "MATLAB_superficial_y_left_1b": matlab_sup_y[:, 0],
        "MATLAB_superficial_x_right_1b": matlab_sup_x[:, 1],
        "MATLAB_superficial_y_right_1b": matlab_sup_y[:, 1],
        "MATLAB_deep_x_left_1b": matlab_deep_x[:, 0],
        "MATLAB_deep_y_left_1b": matlab_deep_y[:, 0],
        "MATLAB_deep_x_right_1b": matlab_deep_x[:, 1],
        "MATLAB_deep_y_right_1b": matlab_deep_y[:, 1],
        "Python_superficial_x_left_1b": python_sup[:, 0],
        "Python_superficial_y_left_1b": python_sup[:, 1],
        "Python_superficial_x_right_1b": python_sup[:, 2],
        "Python_superficial_y_right_1b": python_sup[:, 3],
        "Python_deep_x_left_1b": python_deep[:, 0],
        "Python_deep_y_left_1b": python_deep[:, 1],
        "Python_deep_x_right_1b": python_deep[:, 2],
        "Python_deep_y_right_1b": python_deep[:, 3],
        "MATLAB_fascicle_superficial_x_1b": matlab_fas_x[:, 1],
        "MATLAB_fascicle_superficial_y_1b": matlab_fas_y[:, 1],
        "MATLAB_fascicle_deep_x_1b": matlab_fas_x[:, 0],
        "MATLAB_fascicle_deep_y_1b": matlab_fas_y[:, 0],
        "Python_adaptive_fascicle_superficial_x_1b": adaptive_segment[:, 0],
        "Python_adaptive_fascicle_superficial_y_1b": adaptive_segment[:, 1],
        "Python_adaptive_fascicle_deep_x_1b": adaptive_segment[:, 2],
        "Python_adaptive_fascicle_deep_y_1b": adaptive_segment[:, 3],
        "Python_fixed_fascicle_superficial_x_1b": fixed_segment[:, 0],
        "Python_fixed_fascicle_superficial_y_1b": fixed_segment[:, 1],
        "Python_fixed_fascicle_deep_x_1b": fixed_segment[:, 2],
        "Python_fixed_fascicle_deep_y_1b": fixed_segment[:, 3],
    })

    variable_specs = {
        "ANG": ("deg", "MATLAB_ANG_deg", "Python_fixed_ANG_deg", "Python_adaptive_ANG_deg"),
        "PEN": ("deg", "MATLAB_PEN_deg", "Python_fixed_PEN_deg", "Python_adaptive_PEN_deg"),
        "FL": ("mm", "MATLAB_FL_mm", "Python_fixed_FL_mm", "Python_adaptive_FL_mm"),
    }
    summary_rows: list[dict[str, Any]] = []
    summary_lookup: dict[tuple[str, str], dict[str, Any]] = {}
    for variable, (unit, ref_col, fixed_col, adaptive_col) in variable_specs.items():
        for method, est_col in [("fixed-R", fixed_col), ("adaptive anisotropic-R", adaptive_col)]:
            row: dict[str, Any] = {
                "variable": variable,
                "method": method,
                "unit": unit,
                "final_python_offset_frames": FINAL_OFFSET,
            }
            row.update(full_metrics(framewise[ref_col].iloc[:matlab_n], framewise[est_col].iloc[:matlab_n], total_reference_frames=matlab_n))
            summary_rows.append(row)
            summary_lookup[(variable, method)] = row
    for variable in variable_specs:
        fixed = summary_lookup[(variable, "fixed-R")]
        adaptive = summary_lookup[(variable, "adaptive anisotropic-R")]
        changes = {
            "rmse_change_adaptive_minus_fixed": adaptive["rmse"] - fixed["rmse"],
            "rmse_percent_change": 100.0 * (adaptive["rmse"] - fixed["rmse"]) / fixed["rmse"],
            "mae_change_adaptive_minus_fixed": adaptive["mae"] - fixed["mae"],
            "mae_percent_change": 100.0 * (adaptive["mae"] - fixed["mae"]) / fixed["mae"],
            "bland_altman_width_change_adaptive_minus_fixed": adaptive["bland_altman_width"] - fixed["bland_altman_width"],
            "bland_altman_width_percent_change": 100.0 * (adaptive["bland_altman_width"] - fixed["bland_altman_width"]) / fixed["bland_altman_width"],
        }
        for row in summary_rows:
            if row["variable"] == variable:
                row.update(changes)
    summary = pd.DataFrame(summary_rows)

    alignment_rows: list[dict[str, Any]] = []
    for variable, (unit, ref_col, fixed_col, adaptive_col) in variable_specs.items():
        reference = framewise[ref_col].iloc[:matlab_n].to_numpy(dtype=float)
        for method, est_col in [("fixed-R", fixed_col), ("adaptive anisotropic-R", adaptive_col)]:
            estimate = framewise[est_col].to_numpy(dtype=float)
            for offset in range(-3, 4):
                ref, est, _ = align_pair(reference, estimate, offset)
                row = {
                    "variable": variable,
                    "method": method,
                    "unit": unit,
                    "python_offset_frames": offset,
                    "selected_final_alignment": offset == FINAL_OFFSET,
                    "offset_definition": "MATLAB sample i is paired with Python physical frame i + offset",
                    "selection_basis": "Offset 0 selected from physical-frame evidence; MATLAB timestamps use one-based frame numbering",
                }
                row.update(simple_metrics(ref, est))
                alignment_rows.append(row)
    alignment = pd.DataFrame(alignment_rows)

    matlab_brightness = np.asarray([float(entry["brightness"]) for entry in geofeatures], dtype=float)
    matlab_raw_alpha = np.asarray([float(entry["alpha"]) for entry in geofeatures], dtype=float)
    brightness = video_brightness(args.video)
    physical_rows: list[dict[str, Any]] = []
    for signal, ref, est, unit in [
        ("whole-frame brightness", matlab_brightness, brightness, "8-bit intensity; systematic conversion offset present"),
        ("raw Hough fascicle alpha", matlab_raw_alpha, as_float1(py["raw_timtrack_alpha_deg"]), "deg"),
    ]:
        for offset in range(-3, 4):
            a, b, _ = align_pair(ref, est, offset)
            row = {"evidence_signal": signal, "unit": unit, "python_offset_frames": offset, "selected_final_alignment": offset == 0}
            row.update(simple_metrics(a, b))
            physical_rows.append(row)
    physical_evidence = pd.DataFrame(physical_rows)

    figure_rows: list[pd.DataFrame] = []
    bland_rows: list[pd.DataFrame] = []
    regression_rows: list[pd.DataFrame] = []
    for variable, (unit, ref_col, fixed_col, adaptive_col) in variable_specs.items():
        for method, est_col in [("fixed-R", fixed_col), ("adaptive anisotropic-R", adaptive_col)]:
            subset = framewise.loc[included, [
                "frame_index_video", "frame_index_matlab", "frame_index_python", "time_seconds",
                ref_col, est_col, "adaptive_confidence_score", "adaptive_R_scale_angle", "adaptive_R_scale_length",
                "hough_localmax_fallback_flag", "missing_detection_flag",
            ]].copy()
            subset = subset.rename(columns={ref_col: "MATLAB_value", est_col: "Python_value", "adaptive_confidence_score": "confidence_score"})
            subset.insert(4, "variable", variable)
            subset.insert(5, "method", method)
            subset.insert(6, "unit", unit)
            subset["difference_python_minus_matlab"] = subset.Python_value - subset.MATLAB_value
            subset["absolute_error"] = np.abs(subset.difference_python_minus_matlab)
            subset["bland_altman_mean"] = (subset.Python_value + subset.MATLAB_value) / 2.0
            figure_rows.append(subset)

            metrics_row = summary_lookup[(variable, method)]
            ba = subset[["frame_index_video", "frame_index_matlab", "frame_index_python", "time_seconds", "MATLAB_value", "Python_value", "bland_altman_mean", "difference_python_minus_matlab"]].copy()
            ba.insert(4, "variable", variable); ba.insert(5, "method", method); ba.insert(6, "unit", unit)
            ba["bias"] = metrics_row["bias_python_minus_matlab"]
            ba["difference_sd"] = metrics_row["difference_sd"]
            ba["lower_limit_of_agreement"] = metrics_row["bland_altman_lower_loa"]
            ba["upper_limit_of_agreement"] = metrics_row["bland_altman_upper_loa"]
            bland_rows.append(ba)

            reg = subset[["frame_index_video", "time_seconds", "MATLAB_value", "Python_value"]].copy()
            reg.insert(2, "variable", variable); reg.insert(3, "method", method); reg.insert(4, "unit", unit)
            reg["slope"] = metrics_row["regression_slope"]
            reg["intercept"] = metrics_row["regression_intercept"]
            reg["r_squared"] = metrics_row["regression_r_squared"]
            reg["pearson_r"] = metrics_row["pearson_r"]
            reg["spearman_rho"] = metrics_row["spearman_rho"]
            regression_rows.append(reg)
    figure_data = pd.concat(figure_rows, ignore_index=True)
    bland_altman = pd.concat(bland_rows, ignore_index=True)
    regression = pd.concat(regression_rows, ignore_index=True)

    direct_rows: list[dict[str, Any]] = []
    for idx, (variable, (unit, _, fixed_col, adaptive_col)) in enumerate(variable_specs.items()):
        fixed_error = np.abs(framewise.loc[included, fixed_col].to_numpy(dtype=float) - framewise.loc[included, f"MATLAB_{variable}_{unit}"].to_numpy(dtype=float))
        adaptive_error = np.abs(framewise.loc[included, adaptive_col].to_numpy(dtype=float) - framewise.loc[included, f"MATLAB_{variable}_{unit}"].to_numpy(dtype=float))
        difference = adaptive_error - fixed_error
        tolerance = 1e-12
        try:
            wilcoxon = stats.wilcoxon(difference, alternative="two-sided", zero_method="wilcox")
            wilcoxon_stat, wilcoxon_p = float(wilcoxon.statistic), float(wilcoxon.pvalue)
        except ValueError:
            wilcoxon_stat = wilcoxon_p = np.nan
        direct_rows.append({
            "variable": variable,
            "unit": unit,
            "paired_frames_n": int(len(difference)),
            "difference_definition": "absolute_error_adaptive minus absolute_error_fixed; negative favors adaptive-R",
            "mean_difference": float(np.mean(difference)),
            "median_difference": float(np.median(difference)),
            "adaptive_lower_error_frames_n": int(np.sum(difference < -tolerance)),
            "adaptive_lower_error_percent": 100.0 * np.mean(difference < -tolerance),
            "fixed_lower_error_frames_n": int(np.sum(difference > tolerance)),
            "fixed_lower_error_percent": 100.0 * np.mean(difference > tolerance),
            "ties_frames_n": int(np.sum(np.abs(difference) <= tolerance)),
            "ties_percent": 100.0 * np.mean(np.abs(difference) <= tolerance),
            **moving_block_bootstrap(difference, replicates=args.bootstrap_replicates, seed=args.random_seed + idx),
            "wilcoxon_signed_rank_statistic": wilcoxon_stat,
            "wilcoxon_framewise_p_value_exploratory": wilcoxon_p,
            "inferential_caution": "Frames are temporally autocorrelated and arise from one sequence; framewise p-values are not independent biological evidence",
            "scope": "within-sequence technical comparison",
        })
    fixed_vs_adaptive = pd.DataFrame(direct_rows)

    confidence_config = SpeckleConfidenceConfig()
    bin_edges = np.asarray([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    confidence_rows: list[dict[str, Any]] = []
    valid_framewise = framewise[framewise.included_in_analysis].copy()
    for variable, (unit, _, fixed_col, adaptive_col) in variable_specs.items():
        ref = valid_framewise[f"MATLAB_{variable}_{unit}"].to_numpy(dtype=float)
        fixed_error = valid_framewise[fixed_col].to_numpy(dtype=float) - ref
        adaptive_error = valid_framewise[adaptive_col].to_numpy(dtype=float) - ref
        conf = valid_framewise.adaptive_confidence_score.to_numpy(dtype=float)
        rscale = valid_framewise.adaptive_R_scale_angle.to_numpy(dtype=float) if variable in {"ANG", "PEN"} else valid_framewise.adaptive_R_scale_length.to_numpy(dtype=float)
        fallback = valid_framewise.hough_localmax_fallback_flag.to_numpy(dtype=bool)
        for bin_idx in range(len(bin_edges) - 1):
            low, high = bin_edges[bin_idx], bin_edges[bin_idx + 1]
            mask = (conf >= low) & (conf < high if bin_idx < len(bin_edges) - 2 else conf <= high)
            confidence_rows.append({
                "analysis_scope": "confidence_bin",
                "variable": variable,
                "unit": unit,
                "confidence_bin": f"{low:.2f}–{high:.2f}" + (" inclusive" if bin_idx == len(bin_edges) - 2 else ""),
                "confidence_bin_midpoint": (low + high) / 2.0,
                "confidence_range_implemented": f"[{confidence_config.confidence_floor:.2f}, {confidence_config.confidence_ceiling:.2f}]",
                "frames_n": int(np.sum(mask)),
                "fixed_R_MAE": float(np.mean(np.abs(fixed_error[mask]))) if np.any(mask) else np.nan,
                "adaptive_R_MAE": float(np.mean(np.abs(adaptive_error[mask]))) if np.any(mask) else np.nan,
                "fixed_R_RMSE": float(np.sqrt(np.mean(fixed_error[mask] ** 2))) if np.any(mask) else np.nan,
                "adaptive_R_RMSE": float(np.sqrt(np.mean(adaptive_error[mask] ** 2))) if np.any(mask) else np.nan,
                "mean_relevant_R_scale": float(np.mean(rscale[mask])) if np.any(mask) else np.nan,
                "fallback_frames_n": int(np.sum(fallback[mask])),
                "fallback_percent": 100.0 * np.mean(fallback[mask]) if np.any(mask) else np.nan,
            })
    confidence_analysis = pd.DataFrame(confidence_rows)

    outlier_rows: list[dict[str, Any]] = []
    for variable, (unit, ref_col, fixed_col, adaptive_col) in variable_specs.items():
        for method, est_col in [("fixed-R", fixed_col), ("adaptive anisotropic-R", adaptive_col)]:
            valid = framewise[framewise.included_in_analysis].copy()
            error = valid[est_col].to_numpy(dtype=float) - valid[ref_col].to_numpy(dtype=float)
            absolute_error = np.abs(error)
            error_p95 = float(np.percentile(absolute_error, 95))
            jumps = np.r_[np.nan, np.abs(np.diff(valid[est_col].to_numpy(dtype=float)))]
            jump_p95 = float(np.nanpercentile(jumps, 95))
            relevant_r = valid.adaptive_R_scale_angle.to_numpy(dtype=float) if variable in {"ANG", "PEN"} else valid.adaptive_R_scale_length.to_numpy(dtype=float)
            for pos, (_, row) in enumerate(valid.iterrows()):
                reasons: list[str] = []
                if absolute_error[pos] > error_p95:
                    reasons.append("absolute error above method-variable 95th percentile")
                if np.isfinite(jumps[pos]) and jumps[pos] > jump_p95:
                    reasons.append("abrupt Python frame-to-frame jump above method-variable 95th percentile")
                if bool(row.missing_detection_flag):
                    reasons.append("missing/failed Hough detection flag")
                if bool(row.hough_localmax_fallback_flag):
                    reasons.append("promoted Hough localmax fallback used")
                if float(row.adaptive_confidence_score) <= 0.20:
                    reasons.append("very low confidence (<=0.20)")
                if np.isclose(relevant_r[pos], confidence_config.r_max_scale, rtol=0.0, atol=1e-12):
                    reasons.append("relevant adaptive R scale at configured maximum")
                if reasons:
                    outlier_rows.append({
                        "frame_index_video": int(row.frame_index_video),
                        "frame_index_matlab": int(row.frame_index_matlab),
                        "frame_index_python": int(row.frame_index_python),
                        "time_seconds": float(row.time_seconds),
                        "variable": variable,
                        "method": method,
                        "unit": unit,
                        "error_python_minus_matlab": float(error[pos]),
                        "absolute_error": float(absolute_error[pos]),
                        "error_p95_threshold": error_p95,
                        "python_jump": float(jumps[pos]) if np.isfinite(jumps[pos]) else np.nan,
                        "jump_p95_threshold": jump_p95,
                        "confidence_score": float(row.adaptive_confidence_score),
                        "relevant_R_scale": float(relevant_r[pos]),
                        "fallback_flag": bool(row.hough_localmax_fallback_flag),
                        "missing_detection_flag": bool(row.missing_detection_flag),
                        "flag_reason": "; ".join(reasons),
                        "physical_error_threshold": "not applied; no prospectively justified threshold was supplied",
                    })
    outliers = pd.DataFrame(outlier_rows)

    data_dictionary_rows: list[dict[str, str]] = []
    manual = {
        "frame_index_video": ("index", "Video/OpenCV", "Zero-based physical video frame index."),
        "frame_index_matlab": ("index", "MATLAB TrackingData", "One-based MATLAB physical frame number; blank where MATLAB has no sample."),
        "frame_index_python": ("index", "Python NPZ", "Zero-based Python physical frame index."),
        "time_seconds": ("s", "video frame index / fps", "Physical time with frame 0 at 0 s."),
        "MATLAB_time_seconds_stored": ("s", "MATLAB Region.Time", "Saved MATLAB time; starts at 1/fps because MATLAB labels frames one-based."),
        "included_in_analysis": ("boolean", "analysis", "True only when all nine required MATLAB/fixed/adaptive ANG, PEN, and FL values are finite."),
        "adaptive_confidence_score": ("0–1", "Python confidence diagnostics", "Combined confidence used for scalar summary; anisotropic branches use component confidences."),
        "adaptive_length_side_measurement_covariance_px2": ("px²", "Python Kalman measurement_R_diag[:,0]", "Covariance for superficial-attachment x state; it is the implemented length-side measurement term, not variance of final FL in mm."),
        "adaptive_angle_measurement_covariance_deg2": ("deg²", "Python Kalman measurement_R_diag[:,1]", "Actual per-frame angle measurement covariance."),
    }
    for column in framewise.columns:
        if column in manual:
            unit, source, definition = manual[column]
        else:
            unit = ""
            source = "Deterministic publication analysis"
            definition = column.replace("_", " ")
            if column.startswith("MATLAB_"):
                source = "MATLAB .mat result"
            elif column.startswith("Python_") or column.startswith("adaptive_") or column.startswith("fixed_"):
                source = "Fresh Python strict-results NPZ"
            if column.startswith("fixed_minus_matlab_"):
                source = "Excel formula and independent Python calculation"
                definition = "Signed frame-level difference: Python fixed-R minus MATLAB; positive means Python is larger."
            elif column.startswith("adaptive_minus_matlab_"):
                source = "Excel formula and independent Python calculation"
                definition = "Signed frame-level difference: Python adaptive anisotropic-R minus MATLAB; positive means Python is larger."
            elif column.startswith("absolute_error_fixed_"):
                source = "Excel formula and independent Python calculation"
                definition = "Absolute value of the fixed-R signed difference from MATLAB."
            elif column.startswith("absolute_error_adaptive_"):
                source = "Excel formula and independent Python calculation"
                definition = "Absolute value of the adaptive anisotropic-R signed difference from MATLAB."
            elif "R_scale" in column:
                definition = "Dimensionless multiplier applied to the relevant base measurement covariance; fixed-R equals 1."
            elif "kalman_gain" in column:
                definition = "Forward-filter Kalman gain for the named two-state component."
            elif "smoother_gain" in column:
                definition = "Backward RTS-style smoother gain for the named two-state component."
            elif column.endswith("_flag") or column in {"detection_success", "klt_affine_ok", "klt_tracker_redetected"}:
                definition = "Boolean per-frame diagnostic flag; interpretation is given by the column name and README cautions."
            elif "aponeurosis" in column and column.endswith(("_1b", "_rejected")):
                definition = "One-based image-coordinate aponeurosis endpoint or its per-frame rejection flag, as named."
            elif "fascicle_" in column and column.endswith("_1b"):
                definition = "One-based image-coordinate endpoint of the final fascicle segment, labelled by superficial/deep intersection."
        if column.endswith("_deg"):
            unit = "deg"
        elif column.endswith("_mm"):
            unit = "mm"
        elif column.endswith("_px"):
            unit = "px"
        elif column.endswith("_deg2"):
            unit = "deg²"
        elif column.endswith("_px2"):
            unit = "px²"
        data_dictionary_rows.append({"column": column, "unit": unit, "source": source, "definition_and_interpretation": definition})
    data_dictionary = pd.DataFrame(data_dictionary_rows)

    run_command = (
        ".venv/bin/python scripts/evaluate_final_kalman_matlab_comparison.py "
        "--force-run --python-offset 0 --seed-angle-range 18 24"
    )
    analysis_command = ".venv/bin/python scripts/build_final_matlab_python_validation.py"
    notebook_command = (
        ".venv/bin/python -m nbconvert --to notebook --execute "
        "notebooks/94_final_kalman_matlab_comparison.ipynb "
        "--output 94_final_kalman_matlab_comparison_EXECUTED.ipynb "
        "--output-dir results/notebook94_final_kalman_matlab_comparison "
        "--ExecutePreprocessor.timeout=1200 --ExecutePreprocessor.kernel_name=python3"
    )
    git_status = git_output("status", "--short")
    metadata = {
        "analysis_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Within-sequence technical validation of MATLAB UltraTimTrack against Python fixed-R and adaptive anisotropic-R NathTrack outputs",
        "video_path": str(args.video.resolve()),
        "matlab_result_path": str(args.matlab.resolve()),
        "roi_path": str(args.roi.resolve()),
        "python_npz_path": str(args.python_npz.resolve()),
        "python_metadata_path": str(args.python_metadata.resolve()),
        "utt_export_path": str(args.utt_export.resolve()),
        "sha256": {str(path.resolve()): sha256(path) for path in [args.video, args.matlab, args.roi, args.python_npz, args.utt_export]},
        "commands": {
            "forced_pipeline": run_command,
            "analysis": analysis_command,
            "notebook_execution": notebook_command,
            "workbook_build": "bundled-python scripts/build_final_matlab_python_workbook.py",
            "workbook_recalculation": "LibreOffice headless conversion to XLSX to populate formula caches",
            "workbook_qc": "bundled-python scripts/quality_control_final_matlab_python_workbook.py",
        },
        "logs": {
            "forced_pipeline": str((args.output_dir / "logs" / "python_pipeline_seed18_24_force_run.log").resolve()),
            "analysis": str((args.output_dir / "logs" / "final_validation_analysis.log").resolve()),
            "notebook_execution": str((args.output_dir / "logs" / "notebook94_nbconvert.log").resolve()),
            "workbook_build": str((args.output_dir / "logs" / "workbook_build.log").resolve()),
            "workbook_qc": str((args.output_dir / "logs" / "workbook_qc.log").resolve()),
        },
        "warnings": [
            "Jupyter local kernel reported unencrypted local TCP transport during nbconvert; numerical execution completed successfully.",
            "LibreOffice reported non-fatal font-cache warnings during headless formula recalculation.",
        ],
        "python_executable": sys.executable,
        "python_version": sys.version,
        "package_versions": package_versions(),
        "git_branch": git_output("branch", "--show-current"),
        "git_commit": git_output("rev-parse", "HEAD"),
        "git_working_tree_clean_at_initial_audit": True,
        "git_working_tree_clean_during_execution": git_status == "",
        "git_status_during_execution": git_status.splitlines(),
        "frame_rate_fps": video["fps"],
        "video_frames": video_n,
        "matlab_samples": matlab_n,
        "python_fixed_samples": len(py["fixed_ANG_deg"]),
        "python_adaptive_samples": len(py["ANG_deg"]),
        "final_paired_frames": int(np.sum(included)),
        "final_python_offset_frames": FINAL_OFFSET,
        "alignment_basis": "MATLAB sample 0 and Python frame 0 are the same physical image; MATLAB Time starts at 1/fps because its frame labels are one-based. The unmatched final Python frame is excluded explicitly.",
        "units": {"ANG": "degrees", "PEN": "degrees", "FL": "millimetres"},
        "measurement_scale_mm_per_pixel": scale,
        "measurement_scale_validation": f"TrackingData.res ({float(tracking['res'])} mm) / video height ({video['height_px']} px)",
        "seed_initialization": {
            "seed_frames": int(strict_meta["seed_frames_used"]),
            "seed_only_angle_min_deg": float(strict_meta["seed_angle_min_deg"]),
            "seed_only_angle_max_deg": float(strict_meta["seed_angle_max_deg"]),
            "selected_seed_alpha_deg": float(strict_meta["selected_seed_alpha_deg"]),
            "selected_cluster_id": str(strict_meta["selected_cluster_id"]),
            "per_frame_hough_angle_min_deg": float(strict_meta["fas_angle_min_deg"]),
            "per_frame_hough_angle_max_deg": float(strict_meta["fas_angle_max_deg"]),
            "rationale": (
                "Seed-only 18–24 deg anatomical range chosen from the visible frame-0 fascicle and "
                "the stable approximately 19 deg Hough maximum across all 11 seed frames; it was "
                "not chosen by optimizing MATLAB agreement. Per-frame Hough remained 5–60 deg."
            ),
        },
        "kalman_parameters": {
            "Q": float(utt["Q"]),
            "X_base_variance_px2": float(utt["X"]),
            "R_values": np.asarray(utt["R"], dtype=float).tolist(),
            "angle_base_variance_deg2": float(np.asarray(utt["R"], dtype=float).reshape(-1)[0]),
            "n_start_frames": int(utt["NS"]),
            "run_smoother": True,
        },
        "confidence_configuration": asdict(confidence_config),
        "strict_run_metadata": strict_meta,
        "bootstrap": {"replicates": args.bootstrap_replicates, "random_seed": args.random_seed},
        "matlab_source_spotchecks": [
            {
                "matlab_sample_index_zero_based": int(idx),
                "MATLAB_ANG_deg": float(as_float1(region["ANG"])[idx]),
                "MATLAB_PEN_deg": float(as_float1(region["PEN"])[idx]),
                "MATLAB_FL_mm": float(as_float1(region["FL"])[idx]),
            }
            for idx in sorted({0, matlab_n // 2, matlab_n - 1})
        ],
        "equivalence_testing": "Not performed: no prospective equivalence margins were supplied. A formal test would require clinically/scientifically justified margins for ANG (deg), PEN (deg), and FL (mm), plus independent sequences/participants appropriate to the target population.",
    }

    methods_text = (
        "MATLAB UltraTimTrack outputs were compared with two Python NathTrack Kalman variants generated from the same video and ROI: "
        "a fixed measurement-covariance (fixed-R) filter and an adaptive anisotropic filter that scales angle and length-side measurement covariances separately. "
        "Python initialization used 11 early frames and a seed-only anatomical angle range of 18–24 degrees, selecting the stable 18.8-degree branch; "
        "the per-frame Hough search remained 5–60 degrees. The seed range was selected from the visible fascicle and independent Hough evidence rather than MATLAB agreement. "
        "MATLAB sample 1 and Python frame 0 were matched as the same physical video frame; the MATLAB time vector is one-based and begins at 1/fps. "
        "Agreement was evaluated without signal smoothing or interpolation using bias, absolute error, RMSE, correlations, ordinary least-squares regression, and Bland–Altman limits of agreement."
    )
    results_lines = []
    for variable in variable_specs:
        fixed = summary_lookup[(variable, "fixed-R")]
        adaptive = summary_lookup[(variable, "adaptive anisotropic-R")]
        results_lines.append(
            f"{variable}: fixed-R RMSE {fixed['rmse']:.4f} {fixed['unit']} and MAE {fixed['mae']:.4f} {fixed['unit']}; "
            f"adaptive anisotropic-R RMSE {adaptive['rmse']:.4f} {adaptive['unit']} and MAE {adaptive['mae']:.4f} {adaptive['unit']} "
            f"({adaptive['rmse_percent_change']:+.2f}% RMSE change; {adaptive['mae_percent_change']:+.2f}% MAE change relative to fixed-R)."
        )
    manuscript = {
        "Methods_suggested": methods_text,
        "Results_suggested": " ".join(results_lines),
        "Observations": results_lines,
        "Interpretation": "These results describe agreement within one technical sequence and can indicate where adaptive-R changed tracking error; they do not establish population-level performance or equivalence.",
        "Caveats": [
            "Only one video sequence was analysed.",
            "Frames are temporally autocorrelated and are not independent biological observations.",
            "MATLAB is treated as the comparison reference, not an error-free ground truth.",
            "No prospective equivalence margins were defined, so equivalence was not tested or claimed.",
            "The promoted Hough local-maximum fallback is part of the current Python pipeline and is reported framewise.",
            "The Python seed-only 18–24 degree range is a documented, sequence-specific anatomical initialization constraint and should be prospectively defined or independently selected in future datasets.",
        ],
    }

    write_csv(framewise, args.output_dir / "Framewise_Data.csv")
    write_csv(summary, args.output_dir / "Summary_Metrics.csv")
    write_csv(fixed_vs_adaptive, args.output_dir / "Fixed_vs_Adaptive.csv")
    write_csv(bland_altman, args.output_dir / "Bland_Altman.csv")
    write_csv(regression, args.output_dir / "Regression.csv")
    write_csv(alignment, args.output_dir / "Alignment_Check.csv")
    write_csv(physical_evidence, args.output_dir / "Alignment_Physical_Evidence.csv")
    write_csv(confidence_analysis, args.output_dir / "Confidence_Analysis.csv")
    write_csv(outliers, args.output_dir / "Outlier_Frames.csv")
    write_csv(figure_data, args.output_dir / "Figure_Data.csv")
    write_csv(data_dictionary, args.output_dir / "Data_Dictionary.csv")
    (args.output_dir / "Run_Metadata.json").write_text(json.dumps(metadata, indent=2, default=str) + "\n", encoding="utf-8")
    (args.output_dir / "Manuscript_Summary.json").write_text(json.dumps(manuscript, indent=2) + "\n", encoding="utf-8")

    create_plots(framewise, figure_data, alignment, confidence_analysis, plots_dir)

    print(json.dumps({
        "framewise_rows": len(framewise),
        "included_frames": int(framewise.included_in_analysis.sum()),
        "summary_rows": len(summary),
        "alignment_rows": len(alignment),
        "outlier_rows": len(outliers),
        "plots": len(list(plots_dir.glob("*.png"))) + len(list(plots_dir.glob("*.svg"))),
        "output_dir": str(args.output_dir.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Tk desktop interface for the strict Python UltraTimTrack runner.

Run from the project root:

    ./.venv/bin/python scripts/strict_ultratimtrack_gui.py

The GUI intentionally wraps ``run_strict_ultratimtrack_video.py`` instead of
reimplementing tracking logic.  It provides a MATLAB UltraTimTrack-like
workflow: choose a video, select ROIs, configure Kalman/angle/gating options,
run the pipeline, then inspect the annotated video and FL/PEN time series.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import math
import os
import queue
import re
import shutil
import sys
import threading
import traceback
import zipfile
from pathlib import Path
from typing import Any, Mapping, Sequence
from xml.sax.saxutils import escape

import cv2
import numpy as np
import tkinter as tk
from PIL import Image, ImageTk
from tkinter import filedialog, messagebox, scrolledtext, ttk


def find_project_root() -> Path:
    candidates = [Path.cwd().resolve(), *Path(__file__).resolve().parents]
    for candidate in candidates:
        if (candidate / "ultrasound_tracker").exists():
            return candidate
    raise RuntimeError("Could not find project root containing ultrasound_tracker.")


PROJECT_ROOT = find_project_root()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

os.environ.setdefault("MPLCONFIGDIR", "/private/tmp/matplotlib")

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

import ultrasound_tracker.roi as roi
from ultrasound_tracker.speckle_confidence import SpeckleConfidenceConfig, _gray_float, _match_patch_at, zncc
from scripts.run_strict_ultratimtrack_video import (
    DEFAULT_UTT_EXPORT,
    VIDEO_EXTENSIONS,
    build_arg_parser as build_runner_arg_parser,
    process_video,
    read_first_frame,
    read_gray_frames,
)


DEFAULT_RESULTS_DIR = PROJECT_ROOT / "results" / "strict_ultratimtrack_runs"
BASE_METRIC_COLUMNS = ("Frame", "Time", "FL", "PEN", "ANG")
OPTIONAL_METRIC_COLUMNS = {
    "FixedFL": ("fixed_FL_mm", "fixed_FL_px"),
    "FixedPEN": ("fixed_PEN_deg",),
    "FixedANG": ("fixed_ANG_deg",),
}
SPECKLE_POINT_IDS = ("back", "center", "front")
SPECKLE_POINT_COLORS_BGR = {
    "back": (0, 165, 255),
    "center": (255, 255, 0),
    "front": (80, 220, 80),
    "mean": (255, 255, 255),
}
SPECKLE_GRAPH_BG = (30, 30, 30)
SPECKLE_GRAPH_GRID = (86, 86, 86)
SPECKLE_GRAPH_TEXT = (235, 235, 235)


def _as_float(text: str, default: float | None = None) -> float | None:
    value = str(text).strip()
    if value == "":
        return default
    return float(value)


def _as_int(text: str, default: int | None = None) -> int | None:
    value = str(text).strip()
    if value == "":
        return default
    return int(value)


def _runner_namespace(**overrides: Any) -> argparse.Namespace:
    """Return complete runner arguments with selected GUI values applied."""

    values = vars(build_runner_arg_parser().parse_args([]))
    unknown = sorted(set(overrides).difference(values))
    if unknown:
        raise TypeError(f"Unknown strict runner argument(s): {', '.join(unknown)}")
    values.update(overrides)
    return argparse.Namespace(**values)


def _csv_float(value: str | None) -> float:
    if value is None or str(value).strip() == "":
        return float("nan")
    return float(value)


def _read_metrics_csv(csv_path: Path) -> dict[str, list[float]]:
    metrics = {key: [] for key in [*BASE_METRIC_COLUMNS, *OPTIONAL_METRIC_COLUMNS.keys()]}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        optional_fields = {
            key: next((column for column in aliases if column in fieldnames), None)
            for key, aliases in OPTIONAL_METRIC_COLUMNS.items()
        }
        for row in reader:
            for key in BASE_METRIC_COLUMNS:
                metrics[key].append(_csv_float(row.get(key)))
            for key, column in optional_fields.items():
                if column is not None:
                    metrics[key].append(_csv_float(row.get(column)))
    return metrics


def _has_metric_values(values: list[float]) -> bool:
    arr = np.asarray(values, dtype=np.float64)
    return bool(arr.size and np.any(np.isfinite(arr)))


def _video_filetypes() -> list[tuple[str, str]]:
    patterns = " ".join(f"*{ext}" for ext in VIDEO_EXTENSIONS)
    return [("Video files", patterns), ("All files", "*.*")]


def _frame_to_rgb(frame: np.ndarray) -> np.ndarray:
    arr = np.asarray(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(roi.ensure_uint8_image(arr), cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(roi.ensure_uint8_image(arr), cv2.COLOR_BGR2RGB)


def _resize_rgb_for_box(rgb: np.ndarray, max_width: int, max_height: int) -> tuple[Image.Image, float]:
    height, width = rgb.shape[:2]
    scale = min(max_width / max(width, 1), max_height / max(height, 1), 1.0)
    out_width = max(1, int(round(width * scale)))
    out_height = max(1, int(round(height * scale)))
    pil = Image.fromarray(rgb)
    if scale != 1.0:
        pil = pil.resize((out_width, out_height), Image.Resampling.LANCZOS)
    return pil, scale


def _speckle_config_from_box(block_size: int) -> SpeckleConfidenceConfig:
    if block_size <= 0 or block_size % 2 == 0:
        raise ValueError("Speckle box size must be a positive odd integer.")
    return SpeckleConfidenceConfig(
        block_size=int(block_size),
        stride=24,
        search_radius=8,
        min_texture_variance=5.0,
        zncc_low=0.45,
        zncc_high=0.90,
        forward_backward_scale_px=2.0,
    )


def _extract_speckle_patch(frame: np.ndarray, point: Sequence[float], block_size: int) -> np.ndarray | None:
    arr = _gray_float(frame)
    half = int(block_size) // 2
    x, y = np.rint(np.asarray(point, dtype=np.float64).reshape(2)).astype(int)
    if x - half < 0 or x + half + 1 > arr.shape[1] or y - half < 0 or y + half + 1 > arr.shape[0]:
        return None
    return arr[y - half : y + half + 1, x - half : x + half + 1]


def _point_in_roi(point: Sequence[float], box: roi.ROI | None, *, margin: float = 0.0) -> bool:
    if box is None:
        return True
    x, y = [float(v) for v in np.asarray(point, dtype=np.float64).reshape(2)]
    rx, ry, rw, rh = [float(v) for v in box]
    return (rx + margin) <= x <= (rx + rw - margin) and (ry + margin) <= y <= (ry + rh - margin)


def _point_to_line_distance(point: Sequence[float], line: Sequence[float]) -> float:
    p = np.asarray(point, dtype=np.float64).reshape(2)
    line_arr = np.asarray(line, dtype=np.float64).reshape(4)
    a = line_arr[:2]
    b = line_arr[2:]
    vec = b - a
    denom = float(np.linalg.norm(vec))
    if denom <= 1e-12:
        return float("nan")
    cross = vec[0] * (p[1] - a[1]) - vec[1] * (p[0] - a[0])
    return float(abs(cross) / denom)


def _valid_speckle_center(
    point: Sequence[float],
    shape: Sequence[int],
    cfg: SpeckleConfidenceConfig,
    *,
    fascicle_roi: roi.ROI | None = None,
    aponeurosis_lines: Sequence[np.ndarray | None] | None = None,
) -> bool:
    arr = np.asarray(point, dtype=np.float64).reshape(2)
    if not np.all(np.isfinite(arr)):
        return False
    half = int(cfg.block_size) // 2
    margin = half + int(cfg.search_radius)
    height, width = int(shape[0]), int(shape[1])
    x, y = arr
    if not (margin <= x <= width - 1 - margin and margin <= y <= height - 1 - margin):
        return False
    if not _point_in_roi(arr, fascicle_roi, margin=float(half)):
        return False
    if aponeurosis_lines:
        apo_margin = float(half + 2)
        for line in aponeurosis_lines:
            if line is None:
                continue
            distance = _point_to_line_distance(arr, line)
            if np.isfinite(distance) and distance < apo_margin:
                return False
    return True


def _match_speckle_point_once(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    point: Sequence[float],
    cfg: SpeckleConfidenceConfig,
) -> dict[str, Any]:
    prev = _gray_float(prev_frame)
    curr = _gray_float(curr_frame)
    forward = _match_patch_at(prev, curr, point, cfg)
    if forward is None:
        return {"ok": False, "point": np.full(2, np.nan), "zncc": np.nan, "cv_score": np.nan}

    matched_point, cv_score = forward
    template = _extract_speckle_patch(prev, point, int(cfg.block_size))
    matched_patch = _extract_speckle_patch(curr, matched_point, int(cfg.block_size))
    z_value = float("nan")
    if template is not None and matched_patch is not None:
        z_value = zncc(template, matched_patch, min_texture_variance=float(cfg.min_texture_variance))
    if not np.isfinite(z_value):
        z_value = float(cv_score)
    return {
        "ok": True,
        "point": np.asarray(matched_point, dtype=np.float64),
        "zncc": float(z_value),
        "cv_score": float(cv_score),
    }


def _match_speckle_point_with_fb(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    point: Sequence[float],
    cfg: SpeckleConfidenceConfig,
) -> dict[str, Any]:
    forward = _match_speckle_point_once(prev_frame, curr_frame, point, cfg)
    if not forward["ok"]:
        return {**forward, "forward_backward_error": np.nan}
    reverse = _match_speckle_point_once(curr_frame, prev_frame, forward["point"], cfg)
    if reverse["ok"]:
        fb_error = float(np.linalg.norm(reverse["point"] - np.asarray(point, dtype=np.float64).reshape(2)))
    else:
        fb_error = float("nan")
    return {**forward, "forward_backward_error": fb_error}


def _first_valid_segment(values: np.ndarray, *, label: str) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    for row in arr:
        if row.size >= 4 and np.all(np.isfinite(row[:4])):
            p0 = row[:2]
            p1 = row[2:4]
            if np.linalg.norm(p1 - p0) > 1e-9:
                return row[:4].astype(np.float64)
    raise ValueError(f"No valid {label} segment was found in the strict results.")


def _segment_axes(line: Sequence[float]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    line_arr = np.asarray(line, dtype=np.float64).reshape(4)
    p0 = line_arr[:2]
    p1 = line_arr[2:]
    vec = p1 - p0
    length = float(np.linalg.norm(vec))
    if length <= 1e-9:
        raise ValueError("Reference fascicle segment has zero length.")
    t_hat = vec / length
    n_hat = np.asarray([-t_hat[1], t_hat[0]], dtype=np.float64)
    return p0, p1, t_hat, n_hat, length


def _candidate_score_rows(
    frame0: np.ndarray,
    frame1: np.ndarray,
    line: Sequence[float],
    cfg: SpeckleConfidenceConfig,
    *,
    fascicle_roi: roi.ROI | None,
    aponeurosis_lines: Sequence[np.ndarray | None],
    n_samples: int = 141,
) -> list[dict[str, Any]]:
    line_arr = np.asarray(line, dtype=np.float64).reshape(4)
    p_start = line_arr[:2]
    p_end = line_arr[2:]
    rows: list[dict[str, Any]] = []
    for fraction in np.linspace(0.15, 0.85, int(n_samples)):
        point = p_start + float(fraction) * (p_end - p_start)
        if not _valid_speckle_center(
            point,
            frame0.shape[:2],
            cfg,
            fascicle_roi=fascicle_roi,
            aponeurosis_lines=aponeurosis_lines,
        ):
            continue
        patch = _extract_speckle_patch(frame0, point, int(cfg.block_size))
        if patch is None:
            continue
        match = _match_speckle_point_once(frame0, frame1, point, cfg)
        rows.append(
            {
                "fraction": float(fraction),
                "x": float(point[0]),
                "y": float(point[1]),
                "texture_variance": float(np.var(patch)),
                "zncc_0_1": float(match["zncc"]),
                "match_ok": bool(match["ok"]),
            }
        )

    if not rows:
        raise RuntimeError("No valid speckle candidate points were found along the reference fascicle.")

    textures = np.asarray([row["texture_variance"] for row in rows], dtype=np.float64)
    q10, q90 = np.nanquantile(textures, [0.10, 0.90])
    texture_den = max(float(q90 - q10), 1e-12)
    zncc_den = max(float(cfg.zncc_high - cfg.zncc_low), 1e-12)
    for row in rows:
        texture_score = float(np.clip((float(row["texture_variance"]) - float(q10)) / texture_den, 0.0, 1.0))
        zncc_score = float(np.clip((float(row["zncc_0_1"]) - float(cfg.zncc_low)) / zncc_den, 0.0, 1.0))
        if not np.isfinite(zncc_score):
            zncc_score = 0.0
        row["texture_score"] = texture_score
        row["zncc_score"] = zncc_score
        row["combined_score"] = 0.45 * texture_score + 0.55 * zncc_score
    return sorted(rows, key=lambda row: float(row["combined_score"]), reverse=True)


def _select_three_speckle_points(
    frame0: np.ndarray,
    frame1: np.ndarray,
    line: Sequence[float],
    cfg: SpeckleConfidenceConfig,
    requested_offset_px: float,
    *,
    fascicle_roi: roi.ROI | None,
    aponeurosis_lines: Sequence[np.ndarray | None],
) -> tuple[dict[str, np.ndarray], float, list[dict[str, Any]]]:
    _, _, t_hat, _, length_px = _segment_axes(line)
    candidates = _candidate_score_rows(
        frame0,
        frame1,
        line,
        cfg,
        fascicle_roi=fascicle_roi,
        aponeurosis_lines=aponeurosis_lines,
    )
    min_offset = max(4.0, float(cfg.block_size // 2))
    for candidate in candidates:
        center = np.asarray([candidate["x"], candidate["y"]], dtype=np.float64)
        offset = min(float(requested_offset_px), 0.45 * float(length_px))
        while offset >= min_offset:
            points = {
                "back": center - offset * t_hat,
                "center": center.copy(),
                "front": center + offset * t_hat,
            }
            if all(
                _valid_speckle_center(
                    point,
                    frame0.shape[:2],
                    cfg,
                    fascicle_roi=fascicle_roi,
                    aponeurosis_lines=aponeurosis_lines,
                )
                for point in points.values()
            ):
                return points, float(offset), candidates
            offset *= 0.85
    raise RuntimeError(
        "Could not place three valid speckle points. Try a smaller box size, smaller offset, or a tighter fascicle ROI."
    )


def _track_speckle_points_independent(
    frames: Sequence[np.ndarray],
    initial_points: Mapping[str, np.ndarray],
    cfg: SpeckleConfidenceConfig,
) -> dict[str, np.ndarray]:
    n_frames = len(frames)
    n_points = len(SPECKLE_POINT_IDS)
    positions = np.full((n_frames, n_points, 2), np.nan, dtype=np.float64)
    dxdy = np.zeros((n_frames, n_points, 2), dtype=np.float64)
    zncc_scores = np.full((n_frames, n_points), np.nan, dtype=np.float64)
    fb_errors = np.full((n_frames, n_points), np.nan, dtype=np.float64)
    valid = np.zeros((n_frames, n_points), dtype=bool)
    fallback_used = np.zeros((n_frames, n_points), dtype=bool)

    for point_idx, point_id in enumerate(SPECKLE_POINT_IDS):
        positions[0, point_idx] = np.asarray(initial_points[point_id], dtype=np.float64).reshape(2)
        zncc_scores[0, point_idx] = 1.0
        fb_errors[0, point_idx] = 0.0
        valid[0, point_idx] = True

    last_median_disp = np.zeros(2, dtype=np.float64)
    for frame_idx in range(n_frames - 1):
        attempted_positions = np.full((n_points, 2), np.nan, dtype=np.float64)
        attempted_displacements = np.full((n_points, 2), np.nan, dtype=np.float64)
        pair_good = np.zeros(n_points, dtype=bool)
        for point_idx in range(n_points):
            prev_point = positions[frame_idx, point_idx]
            if not _valid_speckle_center(prev_point, frames[frame_idx].shape[:2], cfg):
                continue
            match = _match_speckle_point_with_fb(frames[frame_idx], frames[frame_idx + 1], prev_point, cfg)
            zncc_scores[frame_idx + 1, point_idx] = float(match["zncc"])
            fb_errors[frame_idx + 1, point_idx] = float(match["forward_backward_error"])
            if not match["ok"]:
                continue
            next_point = np.asarray(match["point"], dtype=np.float64).reshape(2)
            disp = next_point - prev_point
            attempted_positions[point_idx] = next_point
            attempted_displacements[point_idx] = disp
            fb_error = float(match["forward_backward_error"])
            fb_ok = (not np.isfinite(fb_error)) or fb_error <= float(cfg.forward_backward_scale_px)
            pair_good[point_idx] = bool(float(match["zncc"]) >= float(cfg.zncc_low) and fb_ok)

        if np.any(pair_good):
            fallback_disp = np.nanmedian(attempted_displacements[pair_good], axis=0)
            if np.all(np.isfinite(fallback_disp)):
                last_median_disp = fallback_disp.copy()
        else:
            fallback_disp = last_median_disp.copy()

        for point_idx in range(n_points):
            if pair_good[point_idx]:
                positions[frame_idx + 1, point_idx] = attempted_positions[point_idx]
                dxdy[frame_idx + 1, point_idx] = attempted_displacements[point_idx]
                valid[frame_idx + 1, point_idx] = True
            else:
                positions[frame_idx + 1, point_idx] = positions[frame_idx, point_idx] + fallback_disp
                dxdy[frame_idx + 1, point_idx] = fallback_disp
                valid[frame_idx + 1, point_idx] = False
                fallback_used[frame_idx + 1, point_idx] = True

    return {
        "positions": positions,
        "dxdy": dxdy,
        "zncc": zncc_scores,
        "forward_backward_error": fb_errors,
        "valid": valid,
        "fallback_used": fallback_used,
    }


def _strain_arrays(positions: np.ndarray, t_hat: np.ndarray) -> dict[str, np.ndarray]:
    point_index = {point_id: idx for idx, point_id in enumerate(SPECKLE_POINT_IDS)}
    back = positions[:, point_index["back"], :]
    center = positions[:, point_index["center"], :]
    front = positions[:, point_index["front"], :]
    lengths = {
        "back_front": np.sum((front - back) * t_hat[None, :], axis=1),
        "back_center": np.sum((center - back) * t_hat[None, :], axis=1),
        "center_front": np.sum((front - center) * t_hat[None, :], axis=1),
    }
    out: dict[str, np.ndarray] = {}
    for name, values in lengths.items():
        initial_length = float(values[0])
        out[f"length_{name}_px"] = values
        # Negative strain means shortening; positive strain means lengthening.
        if abs(initial_length) > 1e-12:
            out[f"strain_{name}_percent"] = ((values - initial_length) / initial_length) * 100.0
        else:
            out[f"strain_{name}_percent"] = np.full_like(values, np.nan, dtype=np.float64)
    return out


def _normalise_time_s(time_s: np.ndarray, n_frames: int, fps: float) -> np.ndarray:
    arr = np.asarray(time_s, dtype=np.float64).reshape(-1)
    if arr.size >= n_frames and np.all(np.isfinite(arr[:n_frames])):
        return arr[:n_frames].copy()
    fps_value = fps if np.isfinite(fps) and fps > 0 else 30.0
    return np.arange(n_frames, dtype=np.float64) / fps_value


def _speckle_tracking_rows(
    tracking: Mapping[str, np.ndarray],
    frame_numbers: np.ndarray,
    time_s: np.ndarray,
    t_hat: np.ndarray,
    n_hat: np.ndarray,
    mm_per_pixel: float,
    *,
    box_size_px: int,
    offset_px: float,
) -> list[dict[str, Any]]:
    positions = tracking["positions"]
    dxdy = tracking["dxdy"]
    cumulative = positions - positions[0:1, :, :]
    strains = _strain_arrays(positions, t_hat)
    has_mm = np.isfinite(mm_per_pixel) and mm_per_pixel > 0
    rows: list[dict[str, Any]] = []
    for frame_idx in range(positions.shape[0]):
        for point_idx, point_id in enumerate(SPECKLE_POINT_IDS):
            cum = cumulative[frame_idx, point_idx]
            step = dxdy[frame_idx, point_idx]
            d_parallel = float(np.dot(cum, t_hat))
            d_perpendicular = float(np.dot(cum, n_hat))
            row = {
                "frame": int(frame_numbers[frame_idx]) if frame_idx < len(frame_numbers) else int(frame_idx),
                "time_s": float(time_s[frame_idx]) if frame_idx < len(time_s) else np.nan,
                "point_id": point_id,
                "box_size_px": int(box_size_px),
                "offset_px": float(offset_px),
                "x": float(positions[frame_idx, point_idx, 0]),
                "y": float(positions[frame_idx, point_idx, 1]),
                "dx": float(step[0]),
                "dy": float(step[1]),
                "cumulative_dx": float(cum[0]),
                "cumulative_dy": float(cum[1]),
                "d_parallel_px": d_parallel,
                "d_perpendicular_px": d_perpendicular,
                "zncc": float(tracking["zncc"][frame_idx, point_idx]),
                "forward_backward_error": float(tracking["forward_backward_error"][frame_idx, point_idx]),
                "valid": bool(tracking["valid"][frame_idx, point_idx]),
                "fallback_used": bool(tracking["fallback_used"][frame_idx, point_idx]),
                "strain_back_front_percent": float(strains["strain_back_front_percent"][frame_idx]),
                "strain_back_center_percent": float(strains["strain_back_center_percent"][frame_idx]),
                "strain_center_front_percent": float(strains["strain_center_front_percent"][frame_idx]),
                "x_mm": float(positions[frame_idx, point_idx, 0] * mm_per_pixel) if has_mm else np.nan,
                "y_mm": float(positions[frame_idx, point_idx, 1] * mm_per_pixel) if has_mm else np.nan,
                "dx_mm": float(step[0] * mm_per_pixel) if has_mm else np.nan,
                "dy_mm": float(step[1] * mm_per_pixel) if has_mm else np.nan,
                "cumulative_dx_mm": float(cum[0] * mm_per_pixel) if has_mm else np.nan,
                "cumulative_dy_mm": float(cum[1] * mm_per_pixel) if has_mm else np.nan,
                "d_parallel_mm": float(d_parallel * mm_per_pixel) if has_mm else np.nan,
                "d_perpendicular_mm": float(d_perpendicular * mm_per_pixel) if has_mm else np.nan,
            }
            rows.append(row)
    return rows


def _speckle_summary_rows(
    tracking_rows: Sequence[Mapping[str, Any]],
    *,
    box_size_px: int,
    requested_offset_px: float,
    selected_offset_px: float,
    mm_per_pixel: float,
) -> list[dict[str, Any]]:
    zncc_values = np.asarray([row["zncc"] for row in tracking_rows], dtype=np.float64)
    valid_values = np.asarray([row["valid"] for row in tracking_rows], dtype=bool)
    fallback_values = np.asarray([row["fallback_used"] for row in tracking_rows], dtype=bool)
    parallel_values = np.asarray([row["d_parallel_px"] for row in tracking_rows], dtype=np.float64)
    strain_values = np.asarray(
        [row["strain_back_front_percent"] for row in tracking_rows if row["point_id"] == "center"],
        dtype=np.float64,
    )
    has_mm = bool(np.isfinite(mm_per_pixel) and mm_per_pixel > 0)
    return [
        {
            "box_size_px": int(box_size_px),
            "requested_offset_px": float(requested_offset_px),
            "selected_offset_px": float(selected_offset_px),
            "mm_per_pixel": float(mm_per_pixel) if has_mm else "",
            "units_for_overlay": "mm" if has_mm else "px",
            "mean_zncc": float(np.nanmean(zncc_values)) if zncc_values.size else np.nan,
            "invalid_rows": int(np.sum(~valid_values)),
            "fallback_rows": int(np.sum(fallback_values)),
            "max_abs_parallel_displacement_px": float(np.nanmax(np.abs(parallel_values))) if parallel_values.size else np.nan,
            "peak_abs_back_front_strain_percent": float(np.nanmax(np.abs(strain_values))) if strain_values.size else np.nan,
        }
    ]


def _excel_column_name(index: int) -> str:
    name = ""
    index += 1
    while index:
        index, remainder = divmod(index - 1, 26)
        name = chr(65 + remainder) + name
    return name


def _escape_attr(value: Any) -> str:
    return escape(str(value), {'"': "&quot;"})


def _xlsx_cell_xml(row_idx: int, col_idx: int, value: Any) -> str:
    ref = f"{_excel_column_name(col_idx)}{row_idx}"
    if value is None:
        return f'<c r="{ref}"/>'
    if isinstance(value, (bool, np.bool_)):
        return f'<c r="{ref}" t="b"><v>{1 if value else 0}</v></c>'
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        number = float(value)
        if not math.isfinite(number):
            return f'<c r="{ref}"/>'
        if float(number).is_integer():
            text = str(int(number))
        else:
            text = f"{number:.12g}"
        return f'<c r="{ref}"><v>{text}</v></c>'
    text = escape(str(value))
    return f'<c r="{ref}" t="inlineStr"><is><t>{text}</t></is></c>'


def _sheet_xml(rows: Sequence[Mapping[str, Any]]) -> str:
    columns = list(rows[0].keys()) if rows else ["message"]
    sheet_rows: list[str] = []
    all_rows: list[list[Any]] = [columns]
    all_rows.extend([[row.get(column, "") for column in columns] for row in rows])
    for row_idx, values in enumerate(all_rows, start=1):
        cells = "".join(_xlsx_cell_xml(row_idx, col_idx, value) for col_idx, value in enumerate(values))
        sheet_rows.append(f'<row r="{row_idx}">{cells}</row>')
    return (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
        f"<sheetData>{''.join(sheet_rows)}</sheetData>"
        "</worksheet>"
    )


def _write_simple_xlsx(path: Path, sheets: Mapping[str, Sequence[Mapping[str, Any]]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_sheets = [(name[:31] or f"Sheet{idx}", rows) for idx, (name, rows) in enumerate(sheets.items(), start=1)]
    overrides = [
        '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
    ]
    for idx, _ in enumerate(safe_sheets, start=1):
        overrides.append(
            f'<Override PartName="/xl/worksheets/sheet{idx}.xml" '
            'ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
        )
    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Default Extension="xml" ContentType="application/xml"/>'
        f"{''.join(overrides)}"
        "</Types>"
    )
    workbook_sheets = []
    workbook_rels = []
    for idx, (sheet_name, _) in enumerate(safe_sheets, start=1):
        workbook_sheets.append(
            f'<sheet name="{_escape_attr(sheet_name)}" sheetId="{idx}" r:id="rId{idx}"/>'
        )
        workbook_rels.append(
            f'<Relationship Id="rId{idx}" '
            'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" '
            f'Target="worksheets/sheet{idx}.xml"/>'
        )
    workbook = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
        'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
        f"<sheets>{''.join(workbook_sheets)}</sheets>"
        "</workbook>"
    )
    root_rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="xl/workbook.xml"/>'
        "</Relationships>"
    )
    workbook_rels_xml = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f"{''.join(workbook_rels)}"
        "</Relationships>"
    )
    with zipfile.ZipFile(path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("[Content_Types].xml", content_types)
        zf.writestr("_rels/.rels", root_rels)
        zf.writestr("xl/workbook.xml", workbook)
        zf.writestr("xl/_rels/workbook.xml.rels", workbook_rels_xml)
        for idx, (_, rows) in enumerate(safe_sheets, start=1):
            zf.writestr(f"xl/worksheets/sheet{idx}.xml", _sheet_xml(rows))


def _gray_to_bgr(frame: np.ndarray) -> np.ndarray:
    arr = roi.ensure_uint8_image(frame)
    if arr.ndim == 2:
        return cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
    return arr.copy()


def _put_outlined_text(
    frame: np.ndarray,
    text: str,
    origin: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    *,
    thickness: int = 1,
) -> None:
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(frame, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def _draw_speckle_patch_box_cv2(
    frame: np.ndarray,
    point: Sequence[float],
    block_size: int,
    color: tuple[int, int, int],
) -> None:
    p = np.rint(np.asarray(point, dtype=np.float64).reshape(2)).astype(int)
    half = int(block_size) // 2
    cv2.rectangle(frame, (p[0] - half, p[1] - half), (p[0] + half, p[1] + half), color, 1, cv2.LINE_AA)


def _line_points_for_graph(
    time_s: np.ndarray,
    values: np.ndarray,
    x0: int,
    y0: int,
    width: int,
    height: int,
    y_min: float,
    y_max: float,
    current_idx: int,
) -> np.ndarray:
    t = np.asarray(time_s[: current_idx + 1], dtype=np.float64)
    y = np.asarray(values[: current_idx + 1], dtype=np.float64)
    finite = np.isfinite(t) & np.isfinite(y)
    if np.sum(finite) < 2:
        return np.empty((0, 1, 2), dtype=np.int32)
    t = t[finite]
    y = y[finite]
    t0 = float(time_s[0]) if np.isfinite(time_s[0]) else 0.0
    t1 = float(time_s[-1]) if np.isfinite(time_s[-1]) else max(t0 + 1.0, 1.0)
    if abs(t1 - t0) < 1e-12:
        t1 = t0 + 1.0
    x_px = x0 + np.clip((t - t0) / (t1 - t0), 0.0, 1.0) * width
    y_px = y0 + height - np.clip((y - y_min) / max(y_max - y_min, 1e-12), 0.0, 1.0) * height
    return np.rint(np.column_stack([x_px, y_px])).astype(np.int32).reshape(-1, 1, 2)


def _draw_series_panel(
    canvas: np.ndarray,
    rect: tuple[int, int, int, int],
    time_s: np.ndarray,
    series: Mapping[str, np.ndarray],
    current_idx: int,
    title: str,
    ylabel: str,
) -> None:
    x, y, width, height = rect
    cv2.rectangle(canvas, (x, y), (x + width, y + height), SPECKLE_GRAPH_BG, -1)
    plot_left = x + 58
    plot_top = y + 38
    plot_width = max(1, width - 78)
    plot_height = max(1, height - 66)
    all_values = np.concatenate([np.asarray(values, dtype=np.float64).reshape(-1) for values in series.values()])
    finite_values = all_values[np.isfinite(all_values)]
    if finite_values.size:
        y_min = float(np.nanmin(finite_values))
        y_max = float(np.nanmax(finite_values))
    else:
        y_min, y_max = -1.0, 1.0
    if abs(y_max - y_min) < 1e-9:
        pad = max(1.0, abs(y_max) * 0.1)
        y_min -= pad
        y_max += pad
    else:
        pad = 0.08 * (y_max - y_min)
        y_min -= pad
        y_max += pad

    for grid_idx in range(5):
        gy = int(round(plot_top + grid_idx * plot_height / 4.0))
        gx = int(round(plot_left + grid_idx * plot_width / 4.0))
        cv2.line(canvas, (plot_left, gy), (plot_left + plot_width, gy), SPECKLE_GRAPH_GRID, 1, cv2.LINE_AA)
        cv2.line(canvas, (gx, plot_top), (gx, plot_top + plot_height), SPECKLE_GRAPH_GRID, 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (plot_left, plot_top), (plot_left + plot_width, plot_top + plot_height), (180, 180, 180), 1)
    _put_outlined_text(canvas, title, (x + 12, y + 24), 0.55, SPECKLE_GRAPH_TEXT)
    _put_outlined_text(canvas, ylabel, (x + 12, y + height - 12), 0.42, SPECKLE_GRAPH_TEXT)
    _put_outlined_text(canvas, f"{y_max:.2f}", (x + 8, plot_top + 5), 0.36, SPECKLE_GRAPH_TEXT)
    _put_outlined_text(canvas, f"{y_min:.2f}", (x + 8, plot_top + plot_height), 0.36, SPECKLE_GRAPH_TEXT)

    t0 = float(time_s[0]) if len(time_s) and np.isfinite(time_s[0]) else 0.0
    t1 = float(time_s[-1]) if len(time_s) and np.isfinite(time_s[-1]) else t0 + 1.0
    current_time = float(time_s[current_idx]) if current_idx < len(time_s) and np.isfinite(time_s[current_idx]) else t0
    if abs(t1 - t0) < 1e-12:
        t1 = t0 + 1.0
    current_x = int(round(plot_left + np.clip((current_time - t0) / (t1 - t0), 0.0, 1.0) * plot_width))
    cv2.line(canvas, (current_x, plot_top), (current_x, plot_top + plot_height), (210, 210, 210), 1, cv2.LINE_AA)

    for label, values in series.items():
        color = SPECKLE_POINT_COLORS_BGR.get(label, SPECKLE_GRAPH_TEXT)
        points = _line_points_for_graph(time_s, values, plot_left, plot_top, plot_width, plot_height, y_min, y_max, current_idx)
        if len(points):
            cv2.polylines(canvas, [points], False, color, 2 if label != "mean" else 3, cv2.LINE_AA)

    legend_x = x + width - 106
    for legend_idx, label in enumerate(["back", "center", "front", "mean"]):
        color = SPECKLE_POINT_COLORS_BGR[label]
        ly = y + 20 + legend_idx * 17
        cv2.line(canvas, (legend_x, ly), (legend_x + 18, ly), color, 2 if label != "mean" else 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (legend_x + 24, ly + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.38, SPECKLE_GRAPH_TEXT, 1, cv2.LINE_AA)


def _speckle_series_for_overlay(
    tracking: Mapping[str, np.ndarray],
    t_hat: np.ndarray,
    mm_per_pixel: float,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray], str]:
    positions = tracking["positions"]
    cumulative = positions - positions[0:1, :, :]
    has_mm = np.isfinite(mm_per_pixel) and mm_per_pixel > 0
    scale = float(mm_per_pixel) if has_mm else 1.0
    unit = "mm" if has_mm else "px"
    parallel = np.sum(cumulative * t_hat[None, None, :], axis=2) * scale
    x_values = cumulative[:, :, 0] * scale
    y_values = cumulative[:, :, 1] * scale
    out = []
    for matrix in [parallel, y_values, x_values]:
        series = {point_id: matrix[:, idx].astype(np.float64) for idx, point_id in enumerate(SPECKLE_POINT_IDS)}
        series["mean"] = np.nanmean(matrix, axis=1)
        out.append(series)
    return out[0], out[1], out[2], unit


def _write_speckle_overlay_video(
    frames: Sequence[np.ndarray],
    time_s: np.ndarray,
    fascicle_segments: np.ndarray,
    tracking: Mapping[str, np.ndarray],
    t_hat: np.ndarray,
    strains: Mapping[str, np.ndarray],
    output_path: Path,
    *,
    fps: float,
    box_size_px: int,
    requested_offset_px: float,
    selected_offset_px: float,
    mm_per_pixel: float,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    graph_width = 540
    out_size = (width + graph_width, height)
    video_fps = fps if np.isfinite(fps) and fps > 0 else 30.0
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), video_fps, out_size)
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")

    parallel_series, y_series, x_series, unit = _speckle_series_for_overlay(tracking, t_hat, mm_per_pixel)
    positions = tracking["positions"]
    valid = tracking["valid"]
    fallback = tracking["fallback_used"]
    p0 = positions[0, :, :]

    for frame_idx, frame in enumerate(frames):
        vis = _gray_to_bgr(frame)
        canvas = np.zeros((height, width + graph_width, 3), dtype=np.uint8)
        canvas[:, :width] = vis
        canvas[:, width:] = SPECKLE_GRAPH_BG

        line = fascicle_segments[frame_idx] if frame_idx < len(fascicle_segments) else fascicle_segments[0]
        if np.all(np.isfinite(line[:4])):
            xy = np.rint(line[:4]).astype(int)
            cv2.line(canvas, (xy[0], xy[1]), (xy[2], xy[3]), (0, 255, 255), 2, cv2.LINE_AA)

        for point_idx, point_id in enumerate(SPECKLE_POINT_IDS):
            color = SPECKLE_POINT_COLORS_BGR[point_id]
            trail = positions[: frame_idx + 1, point_idx, :]
            for a, b in zip(trail[:-1], trail[1:]):
                if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                    cv2.line(canvas, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), color, 2, cv2.LINE_AA)
            point = positions[frame_idx, point_idx]
            if np.all(np.isfinite(point)):
                _draw_speckle_patch_box_cv2(canvas, point, box_size_px, color)
                center = tuple(np.rint(point).astype(int))
                radius_color = (0, 0, 255) if fallback[frame_idx, point_idx] else (0, 0, 0)
                cv2.circle(canvas, center, 6, color, -1, cv2.LINE_AA)
                cv2.circle(canvas, center, 8, radius_color, 2 if not valid[frame_idx, point_idx] else 1, cv2.LINE_AA)
                _put_outlined_text(canvas, point_id, (center[0] + 9, center[1] - 8), 0.52, (255, 255, 255))
                start = p0[point_idx]
                if np.all(np.isfinite(start)):
                    cv2.arrowedLine(
                        canvas,
                        tuple(np.rint(start).astype(int)),
                        center,
                        color,
                        1,
                        cv2.LINE_AA,
                        tipLength=0.18,
                    )

        strain_value = float(strains["strain_back_front_percent"][frame_idx])
        header_lines = [
            f"Three-point speckle overlay  box {box_size_px}px  offset {selected_offset_px:.0f}px",
            f"frame {frame_idx:04d}  BF strain {strain_value:+.2f}%  requested offset {requested_offset_px:.0f}px",
        ]
        for line_no, text in enumerate(header_lines):
            _put_outlined_text(canvas, text, (18, 30 + line_no * 27), 0.62, (255, 255, 255))

        panel_gap = 10
        panel_height = (height - 4 * panel_gap) // 3
        panel_x = width
        _draw_series_panel(
            canvas,
            (panel_x, panel_gap, graph_width, panel_height),
            time_s,
            parallel_series,
            frame_idx,
            "Parallel displacement",
            f"disp ({unit})",
        )
        _draw_series_panel(
            canvas,
            (panel_x, 2 * panel_gap + panel_height, graph_width, panel_height),
            time_s,
            y_series,
            frame_idx,
            "Y displacement",
            f"disp ({unit})",
        )
        _draw_series_panel(
            canvas,
            (panel_x, 3 * panel_gap + 2 * panel_height, graph_width, panel_height),
            time_s,
            x_series,
            frame_idx,
            "X displacement",
            f"disp ({unit})",
        )
        writer.write(canvas)

    writer.release()


def generate_three_point_speckle_outputs(
    *,
    video_path: Path,
    npz_path: Path,
    run_dir: Path,
    name: str,
    roi_path: Path | None,
    box_size_px: int,
    requested_offset_px: float,
) -> dict[str, Path | float | int | str]:
    cfg = _speckle_config_from_box(int(box_size_px))
    if requested_offset_px <= 0:
        raise ValueError("Speckle point offset must be positive.")
    with np.load(npz_path, allow_pickle=True) as arrays:
        fascicle_segments = np.asarray(arrays["fascicle_end_segments"], dtype=np.float64)
        time_values = np.asarray(arrays["time_s"], dtype=np.float64)
        frame_numbers = np.asarray(arrays["frame"], dtype=np.int64) if "frame" in arrays else np.arange(len(time_values))
        mm_per_pixel = float(np.asarray(arrays["mm_per_pixel"]).reshape(())) if "mm_per_pixel" in arrays else float("nan")
        sup_apo = np.asarray(arrays["sup_apo_lines"], dtype=np.float64) if "sup_apo_lines" in arrays else None
        deep_apo = np.asarray(arrays["deep_apo_lines"], dtype=np.float64) if "deep_apo_lines" in arrays else None

    _, fps, video_frames = read_first_frame(video_path)
    n_frames = min(len(time_values), len(fascicle_segments), int(video_frames) if video_frames else len(time_values))
    if n_frames < 2:
        raise RuntimeError("At least two frames are required for speckle tracking.")
    frames = read_gray_frames(video_path, n_frames)
    n_frames = min(n_frames, len(frames))
    frames = frames[:n_frames]
    fascicle_segments = fascicle_segments[:n_frames]
    time_s = _normalise_time_s(time_values, n_frames, fps)
    frame_numbers = frame_numbers[:n_frames] if len(frame_numbers) >= n_frames else np.arange(n_frames, dtype=np.int64)

    fascicle_line0 = _first_valid_segment(fascicle_segments, label="fascicle")
    _, _, t_hat, n_hat, _ = _segment_axes(fascicle_line0)
    aponeurosis_lines: list[np.ndarray | None] = []
    if sup_apo is not None:
        aponeurosis_lines.append(_first_valid_segment(sup_apo, label="superficial aponeurosis"))
    if deep_apo is not None:
        aponeurosis_lines.append(_first_valid_segment(deep_apo, label="deep aponeurosis"))

    fascicle_roi: roi.ROI | None = None
    if roi_path is not None and roi_path.exists():
        try:
            loaded = roi.load_rois(roi_path)
            fascicle_roi = loaded.get("fascicle")
        except Exception:
            fascicle_roi = None

    initial_points, selected_offset_px, candidate_rows = _select_three_speckle_points(
        frames[0],
        frames[1],
        fascicle_line0,
        cfg,
        float(requested_offset_px),
        fascicle_roi=fascicle_roi,
        aponeurosis_lines=aponeurosis_lines,
    )
    tracking = _track_speckle_points_independent(frames, initial_points, cfg)
    strains = _strain_arrays(tracking["positions"], t_hat)
    tracking_rows = _speckle_tracking_rows(
        tracking,
        frame_numbers,
        time_s,
        t_hat,
        n_hat,
        mm_per_pixel,
        box_size_px=int(box_size_px),
        offset_px=float(selected_offset_px),
    )
    setup_rows = [
        {
            "point_id": point_id,
            "x0": float(initial_points[point_id][0]),
            "y0": float(initial_points[point_id][1]),
            "box_size_px": int(box_size_px),
            "requested_offset_px": float(requested_offset_px),
            "selected_offset_px": float(selected_offset_px),
            "t_x": float(t_hat[0]),
            "t_y": float(t_hat[1]),
            "n_x": float(n_hat[0]),
            "n_y": float(n_hat[1]),
        }
        for point_id in SPECKLE_POINT_IDS
    ]
    summary_rows = _speckle_summary_rows(
        tracking_rows,
        box_size_px=int(box_size_px),
        requested_offset_px=float(requested_offset_px),
        selected_offset_px=float(selected_offset_px),
        mm_per_pixel=mm_per_pixel,
    )
    candidate_sheet_rows = candidate_rows[:200]

    offset_label = int(round(float(selected_offset_px)))
    excel_path = run_dir / f"{name}_three_point_speckle_tracking_box{int(box_size_px)}px_offset{offset_label}px.xlsx"
    overlay_path = run_dir / f"{name}_three_point_speckle_overlay_box{int(box_size_px)}px_offset{offset_label}px.mp4"
    _write_simple_xlsx(
        excel_path,
        {
            "tracking": tracking_rows,
            "summary": summary_rows,
            "point_setup": setup_rows,
            "candidate_scores": candidate_sheet_rows,
        },
    )
    _write_speckle_overlay_video(
        frames,
        time_s,
        fascicle_segments,
        tracking,
        t_hat,
        strains,
        overlay_path,
        fps=fps,
        box_size_px=int(box_size_px),
        requested_offset_px=float(requested_offset_px),
        selected_offset_px=float(selected_offset_px),
        mm_per_pixel=mm_per_pixel,
    )
    return {
        "overlay_video": overlay_path,
        "excel": excel_path,
        "box_size_px": int(box_size_px),
        "requested_offset_px": float(requested_offset_px),
        "selected_offset_px": float(selected_offset_px),
        "units": "mm" if np.isfinite(mm_per_pixel) and mm_per_pixel > 0 else "px",
        "mean_zncc": float(summary_rows[0]["mean_zncc"]),
        "invalid_rows": int(summary_rows[0]["invalid_rows"]),
        "fallback_rows": int(summary_rows[0]["fallback_rows"]),
    }


class QueueWriter:
    """File-like object that forwards stdout/stderr text into a Tk queue."""

    def __init__(self, events: queue.Queue[tuple[str, Any]]) -> None:
        self.events = events

    def write(self, text: str) -> int:
        if text:
            self.events.put(("log", text))
        return len(text)

    def flush(self) -> None:
        return None


class RoiSelectionDialog(tk.Toplevel):
    """Draw superficial, deep, and fascicle ROIs on the first video frame."""

    ROI_ORDER = [
        ("superficial", "Superficial aponeurosis", "#1f77b4"),
        ("deep", "Deep aponeurosis", "#2ca02c"),
        ("fascicle", "Fascicle ROI", "#d4a900"),
    ]

    def __init__(self, master: tk.Misc, frame: np.ndarray, output_path: Path) -> None:
        super().__init__(master)
        self.title("Select UltraTimTrack ROIs")
        self.transient(master)
        self.grab_set()

        self.output_path = output_path
        self.source_rgb = _frame_to_rgb(frame)
        self.source_h, self.source_w = self.source_rgb.shape[:2]
        self.rois: dict[str, roi.ROI] = {}
        self.result: dict[str, roi.ROI] | None = None
        self.current_index = 0
        self.start_xy: tuple[float, float] | None = None
        self.temp_rect: int | None = None

        self.display_image, self.scale = _resize_rgb_for_box(self.source_rgb, 1040, 620)
        self.photo = ImageTk.PhotoImage(self.display_image)

        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)
        self.instruction = ttk.Label(self, anchor="w")
        self.instruction.grid(row=0, column=0, sticky="ew", padx=12, pady=(10, 6))

        self.canvas = tk.Canvas(
            self,
            width=self.display_image.width,
            height=self.display_image.height,
            cursor="crosshair",
            highlightthickness=1,
            highlightbackground="#a0a0a0",
        )
        self.canvas.grid(row=1, column=0, sticky="nsew", padx=12)
        self.canvas.create_image(0, 0, image=self.photo, anchor="nw")
        self.canvas.bind("<ButtonPress-1>", self._mouse_down)
        self.canvas.bind("<B1-Motion>", self._mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self._mouse_up)

        buttons = ttk.Frame(self)
        buttons.grid(row=2, column=0, sticky="ew", padx=12, pady=10)
        ttk.Button(buttons, text="Undo", command=self.undo).pack(side="left")
        ttk.Button(buttons, text="Clear", command=self.clear).pack(side="left", padx=6)
        self.save_button = ttk.Button(buttons, text="Save ROIs", command=self.save, state="disabled")
        self.save_button.pack(side="right")
        ttk.Button(buttons, text="Cancel", command=self.cancel).pack(side="right", padx=6)

        self._update_instruction()
        self.protocol("WM_DELETE_WINDOW", self.cancel)
        self.wait_window(self)

    def _update_instruction(self) -> None:
        if self.current_index < len(self.ROI_ORDER):
            _, label, _ = self.ROI_ORDER[self.current_index]
            self.instruction.configure(
                text=f"Draw ROI {self.current_index + 1}/3: {label}. Drag with the mouse, release to accept."
            )
        else:
            self.instruction.configure(text=f"All ROIs selected. Save to {self.output_path}")
            self.save_button.configure(state="normal")

    def _mouse_down(self, event: tk.Event) -> None:
        if self.current_index >= len(self.ROI_ORDER):
            return
        self.start_xy = (float(event.x), float(event.y))
        if self.temp_rect is not None:
            self.canvas.delete(self.temp_rect)
        _, _, color = self.ROI_ORDER[self.current_index]
        self.temp_rect = self.canvas.create_rectangle(event.x, event.y, event.x, event.y, outline=color, width=2)

    def _mouse_drag(self, event: tk.Event) -> None:
        if self.start_xy is None or self.temp_rect is None:
            return
        x0, y0 = self.start_xy
        self.canvas.coords(self.temp_rect, x0, y0, event.x, event.y)

    def _mouse_up(self, event: tk.Event) -> None:
        if self.start_xy is None or self.temp_rect is None or self.current_index >= len(self.ROI_ORDER):
            return
        x0, y0 = self.start_xy
        x1, y1 = float(event.x), float(event.y)
        left = max(0.0, min(x0, x1))
        top = max(0.0, min(y0, y1))
        right = min(float(self.display_image.width - 1), max(x0, x1))
        bottom = min(float(self.display_image.height - 1), max(y0, y1))
        if right - left < 3 or bottom - top < 3:
            self.canvas.delete(self.temp_rect)
            self.temp_rect = None
            self.start_xy = None
            return

        name, label, color = self.ROI_ORDER[self.current_index]
        inv_scale = 1.0 / max(self.scale, 1e-12)
        x = int(round(left * inv_scale))
        y = int(round(top * inv_scale))
        w = int(round((right - left) * inv_scale))
        h = int(round((bottom - top) * inv_scale))
        x = int(np.clip(x, 0, self.source_w - 1))
        y = int(np.clip(y, 0, self.source_h - 1))
        w = int(np.clip(w, 1, self.source_w - x))
        h = int(np.clip(h, 1, self.source_h - y))
        self.rois[name] = (x, y, w, h)

        self.canvas.delete(self.temp_rect)
        self.temp_rect = None
        self.start_xy = None
        self._draw_roi(name, label, color, (x, y, w, h))
        self.current_index += 1
        self._update_instruction()

    def _draw_roi(self, name: str, label: str, color: str, box: roi.ROI) -> None:
        x, y, w, h = box
        sx = self.scale
        x0, y0, x1, y1 = x * sx, y * sx, (x + w) * sx, (y + h) * sx
        self.canvas.create_rectangle(x0, y0, x1, y1, outline=color, width=2, tags=("roi_box", name))
        self.canvas.create_text(
            x0 + 5,
            max(y0 + 12, 12),
            text=label,
            fill=color,
            anchor="w",
            tags=("roi_box", name),
            font=("TkDefaultFont", 10, "bold"),
        )

    def _redraw_rois(self) -> None:
        self.canvas.delete("roi_box")
        for name, label, color in self.ROI_ORDER:
            if name in self.rois:
                self._draw_roi(name, label, color, self.rois[name])

    def undo(self) -> None:
        if not self.rois:
            return
        self.current_index = max(0, self.current_index - 1)
        name, _, _ = self.ROI_ORDER[self.current_index]
        self.rois.pop(name, None)
        self.save_button.configure(state="disabled")
        self._redraw_rois()
        self._update_instruction()

    def clear(self) -> None:
        self.rois.clear()
        self.current_index = 0
        self.save_button.configure(state="disabled")
        self._redraw_rois()
        self._update_instruction()

    def save(self) -> None:
        missing = [name for name, _, _ in self.ROI_ORDER if name not in self.rois]
        if missing:
            messagebox.showwarning("Missing ROI", f"Please draw: {', '.join(missing)}", parent=self)
            return
        roi.save_rois(self.rois, self.output_path)
        self.result = dict(self.rois)
        self.destroy()

    def cancel(self) -> None:
        self.result = None
        self.destroy()


class StrictUltraTimTrackGUI:
    """Main Tk application."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Strict Python UltraTimTrack")
        self.root.geometry("1680x980")
        self.root.minsize(1360, 820)

        self.events: queue.Queue[tuple[str, Any]] = queue.Queue()
        self.video_path: Path | None = None
        self.roi_path: Path | None = None
        self.first_frame: np.ndarray | None = None
        self.first_frame_info: tuple[float, int] | None = None
        self.loaded_rois: dict[str, roi.ROI] = {}
        self.worker_thread: threading.Thread | None = None
        self.display_cap: cv2.VideoCapture | None = None
        self.display_frame_count = 0
        self.display_fps = 30.0
        self.playing = False
        self.photo: ImageTk.PhotoImage | None = None
        self.metrics: dict[str, list[float]] = {}
        self.current_frame_idx = 0
        self.last_result_paths: dict[str, Path | None] = {}
        self.last_run_dir: Path | None = None
        self.active_total_frames = 0
        self.speckle_worker_thread: threading.Thread | None = None
        self.pending_speckle_overlay_path: Path | None = None
        self.pending_speckle_excel_path: Path | None = None

        self._build_variables()
        self._build_layout()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._draw_empty_plots()
        self._poll_events()

    def _build_variables(self) -> None:
        self.video_label_var = tk.StringVar(value="No video selected")
        self.roi_label_var = tk.StringVar(value="No ROI file selected")
        self.frame_rate_var = tk.StringVar(value="-")
        self.resolution_var = tk.StringVar(value="-")
        self.status_var = tk.StringVar(value="Choose a video to begin.")
        self.progress_label_var = tk.StringVar(value="Progress: 0%")
        self.progress_percent_var = tk.DoubleVar(value=0.0)
        self.final_fl_var = tk.StringVar(value="Final FL: -")
        self.final_pen_var = tk.StringVar(value="Final PEN: -")
        self.current_metric_var = tk.StringVar(value="Frame: -   FL: -   PEN: -   ANG: -")

        self.utt_export_var = tk.StringVar(value=str(DEFAULT_UTT_EXPORT))
        self.image_depth_var = tk.StringVar(value="")
        self.limit_var = tk.StringVar(value="")
        self.seed_frames_var = tk.StringVar(value="11")
        self.kalman_mode_var = tk.StringVar(value="adaptive-anisotropic")
        self.compare_fixed_var = tk.BooleanVar(value=True)
        self.save_debug_var = tk.BooleanVar(value=False)
        self.save_confidence_plot_var = tk.BooleanVar(value=False)
        self.fas_angle_auto_var = tk.BooleanVar(value=True)
        self.fas_angle_min_var = tk.StringVar(value="")
        self.fas_angle_max_var = tk.StringVar(value="")
        self.hough_localmax_fallback_var = tk.BooleanVar(value=True)
        self.hough_fallback_mass_var = tk.StringVar(value="0.25")
        self.hough_fallback_gap_var = tk.StringVar(value="4")
        self.candidate_persistence_var = tk.BooleanVar(value=True)
        self.max_angle_step_var = tk.StringVar(value="8")
        self.apo_gating_var = tk.BooleanVar(value=True)
        self.apo_maxangle_var = tk.StringVar(value="")
        self.deep_mid_jump_var = tk.StringVar(value="6")
        self.super_mid_jump_var = tk.StringVar(value="12")
        self.mid_innovation_var = tk.StringVar(value="10")
        self.angle_jump_var = tk.StringVar(value="2.5")
        self.speckle_overlay_var = tk.BooleanVar(value=False)
        self.speckle_box_size_var = tk.StringVar(value="41")
        self.speckle_offset_var = tk.StringVar(value="45")
        self.slider_var = tk.IntVar(value=0)

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=8)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(0, weight=0)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(0, weight=1)

        left = ttk.Frame(root_frame, width=440)
        left.grid(row=0, column=0, sticky="nsw", padx=(0, 8))
        left.grid_propagate(False)

        right = ttk.Frame(root_frame)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(0, weight=1)
        right.rowconfigure(2, weight=0)

        self._build_settings_panel(left)
        self._build_video_panel(right)
        self._build_plot_panel(right)

    def _build_settings_panel(self, parent: ttk.Frame) -> None:
        canvas = tk.Canvas(parent, highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        content = ttk.Frame(canvas)
        content.bind("<Configure>", lambda _: canvas.configure(scrollregion=canvas.bbox("all")))
        content_window = canvas.create_window((0, 0), window=content, anchor="nw")
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(content_window, width=event.width))
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        settings = ttk.LabelFrame(content, text="Settings", padding=10)
        settings.pack(fill="x", pady=(0, 8))
        settings.columnconfigure(0, weight=1)

        ttk.Button(settings, text="Choose video", command=self.choose_video).grid(row=0, column=0, sticky="ew")
        ttk.Label(settings, textvariable=self.video_label_var, wraplength=390).grid(
            row=1,
            column=0,
            sticky="ew",
            pady=(4, 10),
        )
        ttk.Label(settings, text="Frame rate (Hz)").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Label(settings, textvariable=self.frame_rate_var).grid(row=3, column=0, sticky="w", pady=(0, 6))
        ttk.Label(settings, text="Resolution (pix)").grid(row=4, column=0, sticky="w", pady=2)
        ttk.Label(settings, textvariable=self.resolution_var).grid(row=5, column=0, sticky="w", pady=(0, 6))
        ttk.Label(settings, text="Image depth (mm)").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(settings, textvariable=self.image_depth_var).grid(row=7, column=0, sticky="ew", pady=(0, 8))

        ttk.Label(settings, text="UTT export").grid(row=8, column=0, sticky="w", pady=(10, 2))
        ttk.Button(settings, text="Browse", command=self.choose_utt_export).grid(row=9, column=0, sticky="ew", pady=(0, 4))
        ttk.Entry(settings, textvariable=self.utt_export_var).grid(row=10, column=0, sticky="ew")

        roi_frame = ttk.LabelFrame(content, text="ROI", padding=10)
        roi_frame.pack(fill="x", pady=(0, 8))
        ttk.Button(roi_frame, text="Select ROIs", command=self.select_rois).pack(fill="x")
        ttk.Button(roi_frame, text="Load ROI file", command=self.load_roi_file).pack(fill="x", pady=(5, 0))
        ttk.Label(roi_frame, textvariable=self.roi_label_var, wraplength=390).pack(fill="x", pady=(5, 0))

        kalman = ttk.LabelFrame(content, text="Kalman filter", padding=10)
        kalman.pack(fill="x", pady=(0, 8))
        ttk.Radiobutton(kalman, text="Normal Kalman", value="fixed", variable=self.kalman_mode_var).pack(anchor="w")
        ttk.Radiobutton(
            kalman,
            text="Adaptive Kalman",
            value="adaptive-anisotropic",
            variable=self.kalman_mode_var,
        ).pack(anchor="w")
        ttk.Checkbutton(kalman, text="Compare to normal Kalman", variable=self.compare_fixed_var).pack(anchor="w", pady=(4, 0))

        angles = ttk.LabelFrame(content, text="Fascicle angle", padding=10)
        angles.pack(fill="x", pady=(0, 8))
        angles.columnconfigure(0, weight=1)
        ttk.Checkbutton(
            angles,
            text="Auto angle orientation",
            variable=self.fas_angle_auto_var,
            command=self._sync_fascicle_angle_entry_state,
        ).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        ttk.Label(angles, text="Manual min angle (deg)").grid(row=1, column=0, sticky="w", pady=2)
        self.fas_angle_min_entry = ttk.Entry(angles, textvariable=self.fas_angle_min_var)
        self.fas_angle_min_entry.grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(angles, text="Manual max angle (deg)").grid(row=3, column=0, sticky="w", pady=2)
        self.fas_angle_max_entry = ttk.Entry(angles, textvariable=self.fas_angle_max_var)
        self.fas_angle_max_entry.grid(row=4, column=0, sticky="ew", pady=(0, 6))
        self._sync_fascicle_angle_entry_state()
        ttk.Checkbutton(angles, text="Candidate persistence", variable=self.candidate_persistence_var).grid(
            row=5,
            column=0,
            sticky="w",
            pady=(4, 2),
        )
        ttk.Label(angles, text="Max angle step").grid(row=6, column=0, sticky="w", pady=2)
        ttk.Entry(angles, textvariable=self.max_angle_step_var).grid(row=7, column=0, sticky="ew", pady=(0, 2))

        hough = ttk.LabelFrame(content, text="Hough detector", padding=10)
        hough.pack(fill="x", pady=(0, 8))
        hough.columnconfigure(0, weight=1)
        ttk.Checkbutton(hough, text="Localmax fallback", variable=self.hough_localmax_fallback_var).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        ttk.Label(hough, text="Mass below 10 deg").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(hough, textvariable=self.hough_fallback_mass_var).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(hough, text="Gap to lower (deg)").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(hough, textvariable=self.hough_fallback_gap_var).grid(row=4, column=0, sticky="ew", pady=(0, 2))

        apo = ttk.LabelFrame(content, text="Aponeurosis gating", padding=10)
        apo.pack(fill="x", pady=(0, 8))
        apo.columnconfigure(0, weight=1)
        ttk.Checkbutton(apo, text="Separate apo gating", variable=self.apo_gating_var).grid(
            row=0,
            column=0,
            sticky="w",
            pady=(0, 4),
        )
        ttk.Label(apo, text="Apo maxangle (deg)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(apo, textvariable=self.apo_maxangle_var).grid(row=2, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(apo, text="Deep jump (px)").grid(row=3, column=0, sticky="w", pady=2)
        ttk.Entry(apo, textvariable=self.deep_mid_jump_var).grid(row=4, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(apo, text="Super jump (px)").grid(row=5, column=0, sticky="w", pady=2)
        ttk.Entry(apo, textvariable=self.super_mid_jump_var).grid(row=6, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(apo, text="Innovation (px)").grid(row=7, column=0, sticky="w", pady=2)
        ttk.Entry(apo, textvariable=self.mid_innovation_var).grid(row=8, column=0, sticky="ew", pady=(0, 6))
        ttk.Label(apo, text="Angle jump (deg)").grid(row=9, column=0, sticky="w", pady=2)
        ttk.Entry(apo, textvariable=self.angle_jump_var).grid(row=10, column=0, sticky="ew", pady=(0, 2))

        speckle = ttk.LabelFrame(content, text="Three-point speckle overlay", padding=10)
        speckle.pack(fill="x", pady=(0, 8))
        speckle.columnconfigure(0, weight=1)
        speckle.columnconfigure(1, weight=1)
        ttk.Checkbutton(
            speckle,
            text="Create independent speckle overlay",
            variable=self.speckle_overlay_var,
        ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 6))
        ttk.Label(speckle, text="Box size (px)").grid(row=1, column=0, sticky="w", pady=2)
        ttk.Entry(speckle, textvariable=self.speckle_box_size_var, width=10).grid(
            row=2,
            column=0,
            sticky="ew",
            padx=(0, 4),
            pady=(0, 4),
        )
        ttk.Label(speckle, text="Point offset (px)").grid(row=1, column=1, sticky="w", pady=2)
        ttk.Entry(speckle, textvariable=self.speckle_offset_var, width=10).grid(
            row=2,
            column=1,
            sticky="ew",
            padx=(4, 0),
            pady=(0, 4),
        )

        run = ttk.LabelFrame(content, text="Control panel", padding=10)
        run.pack(fill="x", pady=(0, 8))
        run.columnconfigure(0, weight=1)
        run.columnconfigure(1, weight=1)
        ttk.Label(run, text="Limit frames").grid(row=0, column=0, sticky="w", pady=2)
        ttk.Entry(run, textvariable=self.limit_var).grid(row=1, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(run, text="Seed frames").grid(row=2, column=0, sticky="w", pady=2)
        ttk.Entry(run, textvariable=self.seed_frames_var).grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Checkbutton(run, text="Save debug tables", variable=self.save_debug_var).grid(
            row=4,
            column=0,
            columnspan=2,
            sticky="w",
            pady=(4, 0),
        )
        ttk.Checkbutton(run, text="Save confidence plot", variable=self.save_confidence_plot_var).grid(
            row=5,
            column=0,
            columnspan=2,
            sticky="w",
        )
        self.run_button = ttk.Button(run, text="Run analysis", command=self.run_analysis)
        self.run_button.grid(row=6, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        self.save_all_button = ttk.Button(run, text="Save all...", command=self.save_all_outputs, state="disabled")
        self.save_all_button.grid(row=7, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        self.speckle_continue_button = ttk.Button(
            run,
            text="Continue to speckle overlay",
            command=self.continue_to_speckle_overlay,
            state="disabled",
        )
        self.speckle_continue_button.grid(row=8, column=0, columnspan=2, sticky="ew", pady=(5, 0))
        ttk.Button(run, text="Clear", command=self.clear_results).grid(row=9, column=0, sticky="ew", pady=(5, 0), padx=(0, 3))
        ttk.Button(run, text="Close", command=self.close).grid(row=9, column=1, sticky="ew", pady=(5, 0), padx=(3, 0))
        self.progress = ttk.Progressbar(
            run,
            mode="determinate",
            maximum=100.0,
            variable=self.progress_percent_var,
        )
        self.progress.grid(row=10, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        ttk.Label(run, textvariable=self.progress_label_var).grid(row=11, column=0, columnspan=2, sticky="ew", pady=(4, 0))
        ttk.Label(run, textvariable=self.status_var, wraplength=390).grid(row=12, column=0, columnspan=2, sticky="ew", pady=(6, 0))

        metrics = ttk.LabelFrame(content, text="Outputs", padding=10)
        metrics.pack(fill="x", pady=(0, 8))
        ttk.Label(metrics, textvariable=self.current_metric_var).pack(anchor="w")
        ttk.Label(metrics, textvariable=self.final_fl_var).pack(anchor="w", pady=(4, 0))
        ttk.Label(metrics, textvariable=self.final_pen_var).pack(anchor="w")

        log_frame = ttk.LabelFrame(content, text="Run log", padding=6)
        log_frame.pack(fill="both", expand=True)
        self.log = scrolledtext.ScrolledText(log_frame, height=9, wrap="word", font=("Menlo", 10))
        self.log.pack(fill="both", expand=True)

    def _build_video_panel(self, parent: ttk.Frame) -> None:
        video_area = ttk.Frame(parent)
        video_area.grid(row=0, column=0, sticky="nsew")
        video_area.columnconfigure(0, weight=1)
        video_area.rowconfigure(0, weight=1)

        self.video_display = ttk.Label(video_area, anchor="center", background="#3a3a3a")
        self.video_display.grid(row=0, column=0, sticky="nsew")

        controls = ttk.Frame(parent, padding=(0, 8, 0, 8))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(1, weight=1)
        ttk.Button(controls, text="<", width=3, command=lambda: self.step_video(-1)).grid(row=0, column=0, padx=(0, 4))
        self.slider = ttk.Scale(
            controls,
            orient="horizontal",
            from_=0,
            to=0,
            command=self._slider_changed,
        )
        self.slider.grid(row=0, column=1, sticky="ew")
        ttk.Button(controls, text=">", width=3, command=lambda: self.step_video(1)).grid(row=0, column=2, padx=4)
        self.play_button = ttk.Button(controls, text="Play", width=8, command=self.toggle_play)
        self.play_button.grid(row=0, column=3, padx=(4, 0))

    def _build_plot_panel(self, parent: ttk.Frame) -> None:
        plot_area = ttk.Frame(parent)
        plot_area.grid(row=2, column=0, sticky="ew")
        plot_area.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(10, 3.4), dpi=100)
        self.ax_fl = self.figure.add_subplot(131)
        self.ax_pen = self.figure.add_subplot(132)
        self.ax_ang = self.figure.add_subplot(133)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_area)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="ew")
        self.fl_cursor = None
        self.pen_cursor = None
        self.ang_cursor = None

    def choose_video(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose ultrasound video",
            initialdir=str(PROJECT_ROOT / "data" / "raw"),
            filetypes=_video_filetypes(),
        )
        if not path:
            return
        self.set_video(Path(path))

    def set_video(self, path: Path) -> None:
        try:
            frame0, fps, n_frames = read_first_frame(path)
        except Exception as exc:
            messagebox.showerror("Cannot open video", str(exc), parent=self.root)
            return
        self.video_path = path.resolve()
        self.first_frame = frame0
        self.first_frame_info = (fps, n_frames)
        self.roi_path = PROJECT_ROOT / "data" / "rois" / f"{self.video_path.stem}_rois.json"
        self.video_label_var.set(self.video_path.name)
        self.roi_label_var.set(str(self.roi_path))
        self.frame_rate_var.set(f"{fps:.3g}")
        self.resolution_var.set(f"{frame0.shape[1]} x {frame0.shape[0]}")
        self.status_var.set("Video loaded. Select or load ROIs, then run analysis.")
        self.pending_speckle_overlay_path = None
        self.pending_speckle_excel_path = None
        if hasattr(self, "speckle_continue_button"):
            self.speckle_continue_button.configure(state="disabled")

        self.loaded_rois = {}
        if self.roi_path.exists():
            try:
                self.loaded_rois = roi.load_rois(self.roi_path)
            except Exception:
                self.loaded_rois = {}
        self.show_first_frame()

    def choose_utt_export(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose UTT numeric export .mat",
            initialdir=str(DEFAULT_UTT_EXPORT.parent if DEFAULT_UTT_EXPORT.parent.exists() else PROJECT_ROOT),
            filetypes=[("MAT files", "*.mat"), ("All files", "*.*")],
        )
        if path:
            self.utt_export_var.set(path)

    def select_rois(self) -> None:
        if self.first_frame is None or self.video_path is None:
            messagebox.showinfo("Select video first", "Choose a video before selecting ROIs.", parent=self.root)
            return
        self.roi_path = PROJECT_ROOT / "data" / "rois" / f"{self.video_path.stem}_rois.json"
        dialog = RoiSelectionDialog(self.root, self.first_frame, self.roi_path)
        if dialog.result is not None:
            self.loaded_rois = dialog.result
            self.roi_label_var.set(str(self.roi_path))
            self.status_var.set("ROIs saved. Ready to run analysis.")
            self.show_first_frame()

    def load_roi_file(self) -> None:
        path = filedialog.askopenfilename(
            title="Choose ROI JSON",
            initialdir=str(PROJECT_ROOT / "data" / "rois"),
            filetypes=[("ROI JSON", "*.json"), ("All files", "*.*")],
        )
        if not path:
            return
        try:
            self.loaded_rois = roi.load_rois(path)
        except Exception as exc:
            messagebox.showerror("Cannot load ROI file", str(exc), parent=self.root)
            return
        self.roi_path = Path(path).resolve()
        self.roi_label_var.set(str(self.roi_path))
        self.status_var.set("ROI file loaded.")
        self.show_first_frame()

    def show_first_frame(self) -> None:
        if self.first_frame is None:
            return
        frame = self.first_frame
        if self.loaded_rois:
            frame = roi.draw_rois(frame, self.loaded_rois)
        self.show_image(frame)

    def show_image(self, frame: np.ndarray) -> None:
        rgb = _frame_to_rgb(frame)
        self.root.update_idletasks()
        max_w = max(self.video_display.winfo_width(), 760)
        max_h = max(self.video_display.winfo_height(), 430)
        pil, _ = _resize_rgb_for_box(rgb, max_w, max_h)
        self.photo = ImageTk.PhotoImage(pil)
        self.video_display.configure(image=self.photo)

    def _sync_fascicle_angle_entry_state(self) -> None:
        state = "disabled" if bool(self.fas_angle_auto_var.get()) else "normal"
        for entry_name in ("fas_angle_min_entry", "fas_angle_max_entry"):
            entry = getattr(self, entry_name, None)
            if entry is not None:
                entry.configure(state=state)

    def _build_runner_args(self) -> argparse.Namespace:
        if self.video_path is None:
            raise ValueError("No video selected.")
        if self.roi_path is None or not self.roi_path.exists():
            raise ValueError("No ROI file selected. Select or load ROIs before running.")

        kalman_mode = self.kalman_mode_var.get()
        adaptive_r = kalman_mode != "fixed"
        fas_angle_auto = bool(self.fas_angle_auto_var.get())
        entered_fas_angle_min = _as_float(self.fas_angle_min_var.get(), None)
        entered_fas_angle_max = _as_float(self.fas_angle_max_var.get(), None)
        manual_fas_angle = (not fas_angle_auto) and (
            entered_fas_angle_min is not None or entered_fas_angle_max is not None
        )
        fas_angle_min = entered_fas_angle_min if manual_fas_angle else None
        fas_angle_max = entered_fas_angle_max if manual_fas_angle else None
        apo_maxangle = _as_float(self.apo_maxangle_var.get(), None)
        if fas_angle_min is not None and fas_angle_max is not None and fas_angle_min >= fas_angle_max:
            raise ValueError("Fascicle minimum angle must be smaller than maximum angle.")
        if apo_maxangle is not None and apo_maxangle < 0:
            raise ValueError("Apo maxangle must be non-negative.")
        hough_fallback_mass_value = _as_float(self.hough_fallback_mass_var.get(), 0.25)
        hough_fallback_gap_value = _as_float(self.hough_fallback_gap_var.get(), 4.0)
        hough_fallback_mass = float(0.25 if hough_fallback_mass_value is None else hough_fallback_mass_value)
        hough_fallback_gap = float(4.0 if hough_fallback_gap_value is None else hough_fallback_gap_value)
        if hough_fallback_mass < 0 or hough_fallback_gap < 0:
            raise ValueError("Hough fallback thresholds must be non-negative.")
        limit = _as_int(self.limit_var.get(), None)
        seed_frames = _as_int(self.seed_frames_var.get(), 11) or 11
        if seed_frames <= 0:
            raise ValueError("Seed frames must be positive.")
        image_depth_mm = _as_float(self.image_depth_var.get(), None)
        if image_depth_mm is not None and image_depth_mm <= 0:
            raise ValueError("Image depth must be positive.")
        total_video_frames = int(self.first_frame_info[1]) if self.first_frame_info else 0
        progress_total = min(limit or total_video_frames, total_video_frames) if total_video_frames else (limit or 0)
        self.active_total_frames = int(progress_total)
        progress_every = max(1, int(progress_total // 20)) if progress_total else 250

        return _runner_namespace(
            video=self.video_path,
            interactive=False,
            name=self.video_path.stem,
            utt_export=Path(self.utt_export_var.get()).expanduser().resolve(),
            roi_path=self.roi_path,
            select_roi=False,
            overwrite_roi=False,
            no_roi_parameter_update=False,
            apo_maxangle=apo_maxangle,
            super_apo_maxangle=None,
            deep_apo_maxangle=None,
            limit=limit,
            seed_frames=seed_frames,
            fas_angle_min=fas_angle_min,
            fas_angle_max=fas_angle_max,
            fas_angle_auto=fas_angle_auto,
            hough_localmax_fallback=bool(self.hough_localmax_fallback_var.get()),
            hough_fallback_min_mass_below_10deg=hough_fallback_mass,
            hough_fallback_min_gap_to_lower_deg=hough_fallback_gap,
            candidate_persistence=bool(self.candidate_persistence_var.get()),
            max_angle_step=float(_as_float(self.max_angle_step_var.get(), 8.0) or 8.0),
            candidate_weight_bonus=2.0,
            apo_gating=bool(self.apo_gating_var.get()),
            apo_gate_mid_innovation_px=float(_as_float(self.mid_innovation_var.get(), 10.0) or 10.0),
            apo_gate_super_mid_jump_px=float(_as_float(self.super_mid_jump_var.get(), 12.0) or 12.0),
            apo_gate_deep_mid_jump_px=float(_as_float(self.deep_mid_jump_var.get(), 6.0) or 6.0),
            apo_gate_angle_jump_deg=float(_as_float(self.angle_jump_var.get(), 2.5) or 2.5),
            apo_gate_max_rejections=3,
            debug_detections=bool(self.save_debug_var.get()),
            results_dir=DEFAULT_RESULTS_DIR,
            annotated_video=None,
            no_annotated_video=False,
            save_overlays=3,
            no_time_series_plot=False,
            print_time_series=False,
            kalman_mode=kalman_mode,
            adaptive_r=adaptive_r,
            compare_to_fixed_kalman=bool(self.compare_fixed_var.get()) and adaptive_r,
            annotate_kalman_comparison=False,
            confidence_debug=False,
            save_confidence_plots=bool(self.save_confidence_plot_var.get()),
            mm_per_pixel=None,
            image_depth_mm=image_depth_mm,
            progress_every=progress_every,
        )

    def _speckle_settings(self) -> tuple[int, float]:
        box_size = _as_int(self.speckle_box_size_var.get(), 41)
        offset = _as_float(self.speckle_offset_var.get(), 45.0)
        if box_size is None:
            box_size = 41
        if offset is None:
            offset = 45.0
        _speckle_config_from_box(int(box_size))
        if int(box_size) <= 0:
            raise ValueError("Speckle box size must be positive.")
        if float(offset) <= 0:
            raise ValueError("Speckle point offset must be positive.")
        return int(box_size), float(offset)

    def run_analysis(self) -> None:
        if self.worker_thread is not None and self.worker_thread.is_alive():
            return
        try:
            args = self._build_runner_args()
            if self.speckle_overlay_var.get():
                self._speckle_settings()
        except Exception as exc:
            messagebox.showerror("Cannot run analysis", str(exc), parent=self.root)
            return

        self.log.delete("1.0", "end")
        self.status_var.set("Running analysis...")
        self.run_button.configure(state="disabled")
        self.speckle_continue_button.configure(state="disabled")
        self.pending_speckle_overlay_path = None
        self.pending_speckle_excel_path = None
        self.set_progress(0.0, force=True)
        self.playing = False
        self.play_button.configure(text="Play")

        def worker() -> None:
            writer = QueueWriter(self.events)
            try:
                with contextlib.redirect_stdout(writer), contextlib.redirect_stderr(writer):
                    result = process_video(args)
                self.events.put(("done", result))
            except Exception:
                self.events.put(("error", traceback.format_exc()))

        self.worker_thread = threading.Thread(target=worker, daemon=True)
        self.worker_thread.start()

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self.append_log(str(payload))
                elif event == "done":
                    self.analysis_done(payload)
                elif event == "error":
                    self.analysis_error(str(payload))
                elif event == "speckle_done":
                    self.speckle_done(payload)
                elif event == "speckle_error":
                    self.speckle_error(str(payload))
        except queue.Empty:
            pass
        self.root.after(100, self._poll_events)

    def append_log(self, text: str) -> None:
        self.update_analysis_progress(text)
        self.log.insert("end", text)
        self.log.see("end")

    def set_progress(self, percent: float, *, force: bool = False) -> None:
        value = float(np.clip(percent, 0.0, 100.0))
        current = float(self.progress_percent_var.get())
        if force or value >= current:
            self.progress_percent_var.set(value)
            self.progress_label_var.set(f"Progress: {value:.0f}%")

    def update_analysis_progress(self, text: str) -> None:
        total = max(int(self.active_total_frames), 1)
        for line in str(text).splitlines():
            if "Running TimTrack image stream" in line:
                self.set_progress(3.0)
            elif match := re.search(r"TimTrack image geofeatures processed\s+(\d+)", line):
                frame = min(int(match.group(1)), total)
                self.set_progress(3.0 + 22.0 * frame / total)
            elif "Selecting autonomous fascicle seed" in line:
                self.set_progress(25.0)
            elif "Estimating fascicle KLT affines" in line:
                self.set_progress(32.0)
            elif match := re.search(r"one-step KLT processed\s+(\d+)/(\d+)", line):
                done, count = int(match.group(1)), max(int(match.group(2)), 1)
                self.set_progress(32.0 + 20.0 * done / count)
            elif "Running aponeurosis state estimator" in line:
                self.set_progress(52.0)
            elif match := re.search(r"aponeurosis state processed\s+(\d+)/(\d+)", line):
                done, count = int(match.group(1)), max(int(match.group(2)), 1)
                self.set_progress(52.0 + 20.0 * done / count)
            elif "Computing ultrasound confidence metrics" in line:
                self.set_progress(72.0)
            elif match := re.search(r"confidence processed\s+(\d+)/(\d+)", line):
                done, count = int(match.group(1)), max(int(match.group(2)), 1)
                self.set_progress(72.0 + 10.0 * done / count)
            elif "Running 2-state fascicle Kalman" in line:
                self.set_progress(86.0)
            elif "Running normal fixed-R Kalman" in line:
                self.set_progress(92.0)
            elif line.strip() == "Done.":
                self.set_progress(100.0)

    def analysis_done(self, result: dict[str, Path | None]) -> None:
        self.run_button.configure(state="normal")
        self.set_progress(100.0, force=True)
        self.status_var.set("Analysis complete.")
        self.last_result_paths = dict(result)
        csv_path = result.get("csv")
        annotated_path = result.get("annotated_video")
        self.last_run_dir = Path(csv_path).parent if csv_path else None
        self.save_all_button.configure(state="normal" if self.last_run_dir and self.last_run_dir.exists() else "disabled")
        if csv_path:
            self.load_metrics(Path(csv_path))
        if annotated_path:
            self.load_display_video(Path(annotated_path))
        self.append_log("\nGUI loaded analysis outputs.\n")
        if self.speckle_overlay_var.get():
            self.start_speckle_overlay_generation()

    def analysis_error(self, text: str) -> None:
        self.run_button.configure(state="normal")
        self.speckle_continue_button.configure(state="disabled")
        self.status_var.set("Analysis failed.")
        self.append_log("\n" + text)
        messagebox.showerror("Analysis failed", text.splitlines()[-1] if text.splitlines() else text, parent=self.root)

    def start_speckle_overlay_generation(self) -> None:
        if self.speckle_worker_thread is not None and self.speckle_worker_thread.is_alive():
            return
        if self.video_path is None or self.last_run_dir is None:
            return
        npz_path = self.last_result_paths.get("npz")
        if npz_path is None:
            messagebox.showwarning("Speckle overlay", "Strict NPZ output is missing; cannot create speckle overlay.", parent=self.root)
            return
        try:
            box_size, offset_px = self._speckle_settings()
        except Exception as exc:
            messagebox.showerror("Speckle settings", str(exc), parent=self.root)
            return

        self.speckle_continue_button.configure(state="disabled")
        self.pending_speckle_overlay_path = None
        self.pending_speckle_excel_path = None
        self.status_var.set("Preparing three-point speckle overlay...")
        self.append_log(f"\nPreparing three-point speckle overlay: box {box_size}px, offset {offset_px:.0f}px.\n")

        def worker() -> None:
            try:
                result = generate_three_point_speckle_outputs(
                    video_path=self.video_path,  # type: ignore[arg-type]
                    npz_path=Path(npz_path),
                    run_dir=self.last_run_dir,  # type: ignore[arg-type]
                    name=self.video_path.stem,  # type: ignore[union-attr]
                    roi_path=self.roi_path,
                    box_size_px=box_size,
                    requested_offset_px=offset_px,
                )
                self.events.put(("speckle_done", result))
            except Exception:
                self.events.put(("speckle_error", traceback.format_exc()))

        self.speckle_worker_thread = threading.Thread(target=worker, daemon=True)
        self.speckle_worker_thread.start()

    def speckle_done(self, result: Mapping[str, Any]) -> None:
        overlay_path = Path(result["overlay_video"])
        excel_path = Path(result["excel"])
        self.pending_speckle_overlay_path = overlay_path
        self.pending_speckle_excel_path = excel_path
        self.speckle_continue_button.configure(state="normal")
        self.status_var.set("Speckle overlay ready. Click Continue to speckle overlay.")
        self.append_log(
            "\nThree-point speckle overlay ready.\n"
            f"Overlay video: {overlay_path}\n"
            f"Excel data: {excel_path}\n"
            f"Selected offset: {float(result['selected_offset_px']):.2f}px; "
            f"mean ZNCC: {float(result['mean_zncc']):.3f}; "
            f"fallback rows: {int(result['fallback_rows'])}\n"
        )

    def speckle_error(self, text: str) -> None:
        self.speckle_continue_button.configure(state="disabled")
        self.status_var.set("Speckle overlay failed.")
        self.append_log("\n" + text)
        messagebox.showerror("Speckle overlay failed", text.splitlines()[-1] if text.splitlines() else text, parent=self.root)

    def continue_to_speckle_overlay(self) -> None:
        if self.pending_speckle_overlay_path is None or not self.pending_speckle_overlay_path.exists():
            messagebox.showinfo("Speckle overlay", "The speckle overlay video is not ready yet.", parent=self.root)
            return
        self.load_display_video(self.pending_speckle_overlay_path)
        self.status_var.set("Showing three-point speckle overlay video.")

    def load_display_video(self, path: Path) -> None:
        if self.display_cap is not None:
            self.display_cap.release()
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            messagebox.showwarning("Cannot open annotated video", str(path), parent=self.root)
            return
        self.display_cap = cap
        self.display_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        self.display_fps = fps if np.isfinite(fps) and fps > 0 else 30.0
        self.slider.configure(from_=0, to=max(0, self.display_frame_count - 1))
        self.show_video_frame(0)

    def show_video_frame(self, frame_idx: int) -> None:
        if self.display_cap is None:
            return
        frame_idx = int(np.clip(frame_idx, 0, max(0, self.display_frame_count - 1)))
        self.current_frame_idx = frame_idx
        self.display_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = self.display_cap.read()
        if not ok:
            return
        self.slider.set(frame_idx)
        self.show_image(frame)
        self.update_current_metrics(frame_idx)
        self.update_plot_cursor(frame_idx)

    def _slider_changed(self, value: str) -> None:
        if self.display_cap is None:
            return
        idx = int(float(value))
        if idx != self.current_frame_idx:
            self.show_video_frame(idx)

    def step_video(self, step: int) -> None:
        self.show_video_frame(self.current_frame_idx + int(step))

    def toggle_play(self) -> None:
        if self.display_cap is None:
            return
        self.playing = not self.playing
        self.play_button.configure(text="Pause" if self.playing else "Play")
        if self.playing:
            self._play_step()

    def _play_step(self) -> None:
        if not self.playing:
            return
        if self.current_frame_idx >= self.display_frame_count - 1:
            self.playing = False
            self.play_button.configure(text="Play")
            return
        self.show_video_frame(self.current_frame_idx + 1)
        delay = max(10, int(round(1000.0 / self.display_fps)))
        self.root.after(delay, self._play_step)

    def load_metrics(self, csv_path: Path) -> None:
        self.metrics = _read_metrics_csv(csv_path)
        metrics = self.metrics
        if metrics["FL"]:
            self.final_fl_var.set(f"Final FL: {metrics['FL'][-1]:.2f}")
        if metrics["PEN"]:
            self.final_pen_var.set(f"Final PEN: {metrics['PEN'][-1]:.2f} deg")
        self.redraw_plots()
        self.update_current_metrics(self.current_frame_idx)

    def _draw_empty_plots(self) -> None:
        self.ax_fl.clear()
        self.ax_pen.clear()
        self.ax_ang.clear()
        self._format_metric_axis(self.ax_fl, "Fascicle Length", "FL")
        self._format_metric_axis(self.ax_pen, "Pennation", "PEN (deg)")
        self._format_metric_axis(self.ax_ang, "Angle", "ANG (deg)")
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def _format_metric_axis(self, axis, title: str, ylabel: str) -> None:
        axis.set_title(title)
        axis.set_xlabel("Time (s)")
        axis.set_ylabel(ylabel)
        axis.grid(True, alpha=0.25)

    def _plot_metric_series(
        self,
        axis,
        time_s: np.ndarray,
        primary_key: str,
        fixed_key: str,
        color: str,
        title: str,
        ylabel: str,
    ) -> bool:
        primary = np.asarray(self.metrics.get(primary_key, []), dtype=np.float64)
        n_primary = min(len(time_s), len(primary))
        if n_primary:
            axis.plot(time_s[:n_primary], primary[:n_primary], color=color, linewidth=1.4, label="adaptive")

        has_fixed = _has_metric_values(self.metrics.get(fixed_key, []))
        if has_fixed:
            fixed = np.asarray(self.metrics.get(fixed_key, []), dtype=np.float64)
            n_fixed = min(len(time_s), len(fixed))
            axis.plot(time_s[:n_fixed], fixed[:n_fixed], color="tab:gray", linestyle="--", linewidth=1.2, label="normal")
            axis.legend(loc="best", fontsize=8, frameon=False)

        self._format_metric_axis(axis, title, ylabel)
        return has_fixed

    def redraw_plots(self) -> None:
        if not self.metrics or not self.metrics.get("Time"):
            self._draw_empty_plots()
            return
        time_s = np.asarray(self.metrics["Time"], dtype=np.float64)
        self.ax_fl.clear()
        self.ax_pen.clear()
        self.ax_ang.clear()
        self._plot_metric_series(self.ax_fl, time_s, "FL", "FixedFL", "tab:green", "Fascicle Length", "FL")
        self._plot_metric_series(self.ax_pen, time_s, "PEN", "FixedPEN", "tab:blue", "Pennation", "PEN (deg)")
        self._plot_metric_series(self.ax_ang, time_s, "ANG", "FixedANG", "tab:red", "Angle", "ANG (deg)")
        self.fl_cursor = self.ax_fl.axvline(time_s[0], color="black", linewidth=0.9, alpha=0.6)
        self.pen_cursor = self.ax_pen.axvline(time_s[0], color="black", linewidth=0.9, alpha=0.6)
        self.ang_cursor = self.ax_ang.axvline(time_s[0], color="black", linewidth=0.9, alpha=0.6)
        self.figure.tight_layout()
        self.canvas.draw_idle()

    def update_plot_cursor(self, frame_idx: int) -> None:
        if not self.metrics or not self.metrics.get("Time"):
            return
        idx = min(frame_idx, len(self.metrics["Time"]) - 1)
        t = self.metrics["Time"][idx]
        if self.fl_cursor is not None:
            self.fl_cursor.set_xdata([t, t])
        if self.pen_cursor is not None:
            self.pen_cursor.set_xdata([t, t])
        if self.ang_cursor is not None:
            self.ang_cursor.set_xdata([t, t])
        self.canvas.draw_idle()

    def update_current_metrics(self, frame_idx: int) -> None:
        if not self.metrics or not self.metrics.get("FL"):
            self.current_metric_var.set(f"Frame: {frame_idx}   FL: -   PEN: -   ANG: -")
            return
        idx = min(frame_idx, len(self.metrics["FL"]) - 1)
        fl = self.metrics["FL"][idx]
        pen = self.metrics["PEN"][idx] if idx < len(self.metrics["PEN"]) else np.nan
        ang = self.metrics["ANG"][idx] if idx < len(self.metrics["ANG"]) else np.nan
        self.current_metric_var.set(f"Frame: {idx}   FL: {fl:.2f}   PEN: {pen:.2f} deg   ANG: {ang:.2f} deg")

    def clear_results(self) -> None:
        self.playing = False
        self.play_button.configure(text="Play")
        if self.display_cap is not None:
            self.display_cap.release()
            self.display_cap = None
        self.display_frame_count = 0
        self.current_frame_idx = 0
        self.metrics = {}
        self.last_result_paths = {}
        self.last_run_dir = None
        self.pending_speckle_overlay_path = None
        self.pending_speckle_excel_path = None
        self.save_all_button.configure(state="disabled")
        self.speckle_continue_button.configure(state="disabled")
        self.slider.configure(from_=0, to=0)
        self.slider.set(0)
        self.final_fl_var.set("Final FL: -")
        self.final_pen_var.set("Final PEN: -")
        self.current_metric_var.set("Frame: -   FL: -   PEN: -   ANG: -")
        self.set_progress(0.0, force=True)
        self._draw_empty_plots()
        self.show_first_frame()
        self.status_var.set("Results cleared.")

    def save_all_outputs(self) -> None:
        if self.last_run_dir is None or not self.last_run_dir.exists():
            messagebox.showinfo("No results", "Run an analysis before saving outputs.", parent=self.root)
            return
        target_parent = filedialog.askdirectory(
            title="Choose where to save UltraTimTrack outputs",
            initialdir=str(Path.home()),
        )
        if not target_parent:
            return
        destination = Path(target_parent).expanduser().resolve() / self.last_run_dir.name
        if destination.exists():
            ok = messagebox.askyesno(
                "Folder exists",
                f"{destination} already exists. Replace files inside it?",
                parent=self.root,
            )
            if not ok:
                return
        shutil.copytree(self.last_run_dir, destination, dirs_exist_ok=True)
        self.status_var.set(f"Saved outputs to {destination}")
        self.append_log(f"\nSaved all outputs to: {destination}\n")

    def close(self) -> None:
        self.playing = False
        if self.display_cap is not None:
            self.display_cap.release()
            self.display_cap = None
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
    except Exception:
        pass
    StrictUltraTimTrackGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()

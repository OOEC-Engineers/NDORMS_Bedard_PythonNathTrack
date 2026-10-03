#!/usr/bin/env python3
"""Run the strict Python UltraTimTrack-style pipeline on a video.

Examples
--------
Interactive:
    python scripts/run_strict_ultratimtrack_video.py --interactive

Command-line:
    python scripts/run_strict_ultratimtrack_video.py data/raw/UltraTimTrack_test.mp4 \
        --roi-path data/rois/UltraTimTrack_test_rois.json \
        --utt-export data/matlab/UTT_numeric_export.mat
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import cv2
import numpy as np
from scipy.io import loadmat


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

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import ultrasound_tracker.roi as roi
import ultrasound_tracker.utils as ut
from ultrasound_tracker.matlab_aponeurosis import make_matlab_apox
from ultrasound_tracker.matlab_timtrack import (
    detect_timtrack_geofeature_from_image,
    run_timtrack_geofeatures_from_video,
)
from ultrasound_tracker.speckle_confidence import (
    SpeckleConfidenceConfig,
    anisotropic_confidence_to_r_scales,
    combine_anisotropic_confidence_metrics,
    combine_confidence_metrics,
    compute_feature_detection_reliability,
    compute_geometry_stability,
    compute_motion_consistency,
    compute_speckle_coherence,
    confidence_to_r_scale,
)
from ultrasound_tracker.strict_fascicle_seed import (
    FascicleSeedScoringConfig,
    cluster_seed_candidates,
    extract_fascicle_seed_candidates,
    select_autonomous_fascicle_seed,
)
from ultrasound_tracker.ultratrack_klt import (
    UltraTrackKLTConfig,
    run_persistent_affine_video,
)
from ultrasound_tracker.ultratimtrack_aponeurosis import (
    AponeurosisGatingConfig,
    run_matlab_aponeurosis_state_video,
)
from ultrasound_tracker.ultratimtrack_matlab_2state import (
    MatlabTwoStateKalmanConfig,
    run_matlab_2state_kalman,
)


VIDEO_EXTENSIONS = [".avi", ".AVI", ".mp4", ".MP4", ".mov", ".MOV", ".mkv", ".MKV"]
KALMAN_MODES = ("fixed", "adaptive-scalar", "adaptive-anisotropic")
DEFAULT_UTT_EXPORT = PROJECT_ROOT / "data" / "matlab" / "UTT_numeric_export.mat"
LEGACY_UTT_EXPORT = Path("/Users/grosbedou/Documents/GitHub/UltraTimTrack/UTT_numeric_export.mat")
AUTO_ORIENTATION_SIGN_TIE_TOL = 0.02


def finite_positive_float(value: Any) -> float | None:
    """Return ``value`` as a positive finite float, or ``None`` if unavailable."""

    try:
        scalar = float(np.asarray(value, dtype=np.float64).reshape(-1)[0])
    except Exception:
        return None
    if np.isfinite(scalar) and scalar > 0:
        return scalar
    return None


def resolve_measurement_pixel_scale(
    mat_root: Mapping[str, Any],
    *,
    image_height_px: int,
    mm_per_pixel: float | None,
    image_depth_mm: float | None,
) -> dict[str, float | str | None]:
    """Resolve the user-controlled measurement pixel scale.

    The autonomous fascicle seed selector is depth-independent. This scale is
    only used after detection to report lengths/displacements in millimetres.
    """

    height = float(image_height_px)
    matlab_image_depth_mm = finite_positive_float(mat_root.get("ID"))
    direct_mm_per_px = finite_positive_float(mm_per_pixel)
    entered_depth_mm = finite_positive_float(image_depth_mm)
    if direct_mm_per_px is not None:
        measurement_mm_per_px = float(direct_mm_per_px)
        measurement_scale_source = "direct_mm_per_pixel"
    elif entered_depth_mm is not None:
        measurement_mm_per_px = float(entered_depth_mm) / height
        measurement_scale_source = "entered_image_depth_mm"
    else:
        measurement_mm_per_px = float("nan")
        measurement_scale_source = "missing"

    return {
        "mm_per_pixel": measurement_mm_per_px,
        "measurement_mm_per_pixel": measurement_mm_per_px,
        "measurement_scale_source": measurement_scale_source,
        "matlab_export_image_depth_mm": matlab_image_depth_mm,
    }


@dataclass(frozen=True)
class FascicleCandidatePersistenceConfig:
    """Frame-to-frame fascicle candidate persistence settings."""

    enabled: bool = False
    angle_min_deg: float = 5.0
    angle_max_deg: float = 60.0
    max_angle_step_deg: float = 8.0
    hough_weight_bonus_deg: float = 2.0


def kalman_mode_uses_confidence(mode: str) -> bool:
    """Return whether a Kalman mode needs per-frame confidence metrics."""

    return mode in {"adaptive-scalar", "adaptive-anisotropic"}


def list_videos_in_raw(raw_dir: Path) -> list[Path]:
    videos: list[Path] = []
    if raw_dir.exists():
        for ext in VIDEO_EXTENSIONS:
            videos.extend(raw_dir.glob(f"*{ext}"))
    return sorted(set(videos), key=lambda p: p.name.lower())


def prompt_video_from_raw(raw_dir: Path) -> Path:
    videos = list_videos_in_raw(raw_dir)
    if not videos:
        raise FileNotFoundError(f"No video found in {raw_dir}.")

    print("\nVideos found in data/raw:")
    for idx, path in enumerate(videos, start=1):
        print(f"  {idx}. {path.name}")

    while True:
        answer = input("\nVideo to analyze (number or filename): ").strip()
        if answer.isdigit() and 1 <= int(answer) <= len(videos):
            return videos[int(answer) - 1]
        candidate = raw_dir / answer
        if candidate.exists():
            return candidate
        matches = [p for p in videos if p.name.lower() == answer.lower() or p.stem.lower() == answer.lower()]
        if len(matches) == 1:
            return matches[0]
        print("Could not find that video.")


def prompt_yes_no(question: str, default: bool = True) -> bool:
    suffix = " [Y/n]: " if default else " [y/N]: "
    while True:
        answer = input(question + suffix).strip().lower()
        if answer == "":
            return default
        if answer in {"y", "yes", "o", "oui"}:
            return True
        if answer in {"n", "no", "non"}:
            return False
        print("Please answer yes/no.")


def prompt_kalman_mode(default: str = "fixed") -> str:
    """Prompt for the strict runner Kalman mode."""

    aliases = {
        "normal": "fixed",
        "fixed": "fixed",
        "n": "fixed",
        "scalar": "adaptive-scalar",
        "adaptive-scalar": "adaptive-scalar",
        "s": "adaptive-scalar",
        "adaptive": "adaptive-anisotropic",
        "anisotropic": "adaptive-anisotropic",
        "adaptive-anisotropic": "adaptive-anisotropic",
        "a": "adaptive-anisotropic",
    }
    default = aliases.get(str(default).strip().lower(), "fixed")

    print("\nKalman filter mode:")
    print("  1. normal fixed-R Kalman")
    print("  2. scalar adaptive-R Kalman")
    print("  3. anisotropic adaptive-R Kalman")
    while True:
        answer = input(f"Choose Kalman mode [default: {default}]: ").strip().lower()
        if answer == "":
            return default
        if answer in {"1", "2", "3"}:
            return KALMAN_MODES[int(answer) - 1]
        if answer in aliases:
            return aliases[answer]
        print("Please enter 1, 2, 3, normal, scalar, or anisotropic.")


def read_first_frame(video_path: Path) -> tuple[np.ndarray, float, int]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Could not read first frame from {video_path}")
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    return gray, fps, n_frames


def read_gray_frames(video_path: Path, n_frames: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frames: list[np.ndarray] = []
    for _ in range(int(n_frames)):
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy())
    cap.release()
    if not frames:
        raise RuntimeError(f"No frames read from {video_path}")
    return frames


def select_or_load_rois(
    frame0_gray: np.ndarray,
    roi_path: Path,
    *,
    select_roi: bool,
    overwrite_roi: bool,
) -> dict[str, roi.ROI]:
    if roi_path.exists() and not select_roi:
        print(f"Loading ROIs: {roi_path}")
        return roi.load_rois(roi_path)
    if roi_path.exists() and select_roi and not overwrite_roi:
        raise FileExistsError(f"ROI file already exists: {roi_path}. Use --overwrite-roi to replace it.")

    print("\nSelect ROIs in this order:")
    print("  1. superficial aponeurosis")
    print("  2. deep aponeurosis")
    print("  3. fascicle")
    rois = roi.select_all_rois_cv2(frame0_gray, include_fascicle_roi=True)
    roi.save_rois(rois, roi_path)
    print(f"Saved ROIs: {roi_path}")
    return rois


def ellipse_mask_from_roi(shape: tuple[int, int], fascicle_roi: roi.ROI) -> tuple[np.ndarray, np.ndarray]:
    height, width = map(int, shape)
    x, y, w, h = map(float, fascicle_roi)
    rx = max(w / 2.0, 1.0)
    ry = max(h / 2.0, 1.0)
    cx = x + rx + 1.0
    cy = y + ry + 1.0
    yy, xx = np.mgrid[1 : height + 1, 1 : width + 1]
    mask = ((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0
    return mask.astype(bool), np.asarray([ry, rx], dtype=np.float64)


def update_parms_from_rois(parms: dict, rois: Mapping[str, roi.ROI], frame_shape: tuple[int, int]) -> dict:
    """Update MATLAB-like parameter cuts and fascicle Emask from selected ROIs."""

    out = copy.deepcopy(parms)
    height, width = map(int, frame_shape)
    if "apo" not in out:
        out["apo"] = {}
    if "fas" not in out:
        out["fas"] = {}
    out["apo"]["apox"] = make_matlab_apox(width)

    for name, key in [("superficial", "super"), ("deep", "deep")]:
        if name in rois:
            _, y, _, h = rois[name]
            out["apo"].setdefault(key, {})
            out["apo"][key]["cut"] = np.asarray([y / height, (y + h) / height], dtype=np.float64)

    if "fascicle" in rois:
        emask, radius = ellipse_mask_from_roi((height, width), rois["fascicle"])
        out["fas"]["Emask"] = emask
        out["fas"]["Emask_radius"] = radius
        out["fas"]["redo_ROI"] = 0

    return out


def apply_aponeurosis_maxangle_overrides(
    parms: dict,
    *,
    apo_maxangle: float | None = None,
    super_apo_maxangle: float | None = None,
    deep_apo_maxangle: float | None = None,
) -> dict:
    """Override aponeurosis fit angle limits in-place on a copied parms dict."""

    updates = {
        "super": super_apo_maxangle if super_apo_maxangle is not None else apo_maxangle,
        "deep": deep_apo_maxangle if deep_apo_maxangle is not None else apo_maxangle,
    }
    if not any(value is not None for value in updates.values()):
        return parms

    apo = parms.setdefault("apo", {})
    for key, value in updates.items():
        if value is None:
            continue
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("Aponeurosis maxangle overrides must be finite and non-negative.")
        apo.setdefault(key, {})
        apo[key]["fit_method"] = "enforce_maxangle"
        apo[key]["maxangle"] = value
    return parms


def apply_fascicle_angle_overrides(
    parms: dict,
    *,
    fas_angle_min: float | None = None,
    fas_angle_max: float | None = None,
) -> dict:
    """Override TimTrack fascicle Hough angle range in-place on a copied parms dict."""

    if fas_angle_min is None and fas_angle_max is None:
        return parms

    fas = parms.setdefault("fas", {})
    current = np.asarray(fas.get("range", [5.0, 60.0]), dtype=np.float64).reshape(-1)
    if current.size < 2:
        current = np.asarray([5.0, 60.0], dtype=np.float64)
    low = float(current[0] if fas_angle_min is None else fas_angle_min)
    high = float(current[1] if fas_angle_max is None else fas_angle_max)
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        raise ValueError("Fascicle angle range must be finite with min < max.")
    fas["range"] = np.asarray([low, high], dtype=np.float64)
    return parms


def current_fascicle_angle_range(parms: Mapping) -> tuple[float, float]:
    """Return the configured TimTrack fascicle angle range."""

    fas = parms.get("fas", {}) if isinstance(parms, Mapping) else {}
    current = np.asarray(fas.get("range", [5.0, 60.0]), dtype=np.float64).reshape(-1)
    if current.size < 2:
        current = np.asarray([5.0, 60.0], dtype=np.float64)
    low = float(current[0])
    high = float(current[1])
    if not np.isfinite(low) or not np.isfinite(high) or low >= high:
        return 5.0, 60.0
    return low, high


def fascicle_angle_abs_bounds(parms: Mapping) -> tuple[float, float]:
    """Return positive angle magnitudes used to test both fascicle orientations."""

    low, high = current_fascicle_angle_range(parms)
    magnitudes = np.abs(np.asarray([low, high], dtype=np.float64))
    magnitudes = magnitudes[np.isfinite(magnitudes) & (magnitudes > 0)]
    if magnitudes.size < 2:
        return 5.0, 60.0
    low_abs = float(np.min(magnitudes))
    high_abs = float(np.max(magnitudes))
    if low_abs >= high_abs:
        return 5.0, 60.0
    return low_abs, high_abs


def candidate_signed_fascicle_angle_ranges(parms: Mapping) -> list[tuple[float, float]]:
    """Return positive and reverse-signed versions of the configured fascicle range."""

    current_low, current_high = current_fascicle_angle_range(parms)
    low_abs, high_abs = fascicle_angle_abs_bounds(parms)
    positive = (low_abs, high_abs)
    reverse = (-high_abs, -low_abs)
    ordered = [reverse, positive] if current_high <= 0 else [positive, reverse]
    ranges: list[tuple[float, float]] = []
    for low, high in ordered:
        pair = (float(low), float(high))
        if pair not in ranges:
            ranges.append(pair)
    if (current_low, current_high) not in ranges and current_low < current_high:
        ranges.insert(0, (current_low, current_high))
    return ranges


def detect_seed_geofeature_entries(
    seed_frames: Sequence[np.ndarray],
    parms: Mapping,
    *,
    hough_localmax_fallback: bool = False,
    hough_fallback_min_mass_below_10deg: float = 0.25,
    hough_fallback_min_gap_to_lower_deg: float = 4.0,
) -> list[dict[str, Any]]:
    """Run the TimTrack image detector on the frames used for seed selection."""

    entries: list[dict[str, Any]] = []
    for idx, gray in enumerate(seed_frames):
        entry = detect_timtrack_geofeature_from_image(
            gray,
            parms,
            emask_mode="matlab",
            hough_localmax_fallback=bool(hough_localmax_fallback),
            hough_fallback_min_mass_below_10deg=float(hough_fallback_min_mass_below_10deg),
            hough_fallback_min_gap_to_lower_deg=float(hough_fallback_min_gap_to_lower_deg),
        )
        entry["frame"] = idx
        entries.append(entry)
    return entries


def score_fascicle_angle_range(
    seed_frames: Sequence[np.ndarray],
    parms: Mapping,
    *,
    angle_min_deg: float,
    angle_max_deg: float,
    mm_per_px: float | None = None,
    hough_localmax_fallback: bool = False,
    hough_fallback_min_mass_below_10deg: float = 0.25,
    hough_fallback_min_gap_to_lower_deg: float = 4.0,
) -> dict[str, Any]:
    """Score one signed fascicle angle range from early-frame seed stability."""

    test_parms = apply_fascicle_angle_overrides(
        copy.deepcopy(dict(parms)),
        fas_angle_min=float(angle_min_deg),
        fas_angle_max=float(angle_max_deg),
    )
    seed_entries = detect_seed_geofeature_entries(
        seed_frames,
        test_parms,
        hough_localmax_fallback=bool(hough_localmax_fallback),
        hough_fallback_min_mass_below_10deg=float(hough_fallback_min_mass_below_10deg),
        hough_fallback_min_gap_to_lower_deg=float(hough_fallback_min_gap_to_lower_deg),
    )
    seed_config = replace(
        FascicleSeedScoringConfig(min_cluster_frame_coverage=min(8, len(seed_frames))),
        angle_min_deg=float(angle_min_deg),
        angle_max_deg=float(angle_max_deg),
    )
    candidates = extract_fascicle_seed_candidates(seed_entries, seed_frames, mm_per_px=mm_per_px, config=seed_config)
    candidates, clusters = cluster_seed_candidates(candidates, min_frame_coverage=seed_config.min_cluster_frame_coverage)
    if clusters.empty:
        score = -np.inf
        top_cluster: dict[str, Any] = {}
    else:
        top_cluster = clusters.iloc[0].to_dict()
        score = float(top_cluster["cluster_score"])

    return {
        "angle_min_deg": float(angle_min_deg),
        "angle_max_deg": float(angle_max_deg),
        "score": score,
        "top_cluster": top_cluster,
        "parms": test_parms,
        "seed_entries": seed_entries,
        "seed_config": seed_config,
        "candidates": candidates,
        "clusters": clusters,
        "error": None,
    }


def choose_fascicle_angle_range_from_seed_frames(
    seed_frames: Sequence[np.ndarray],
    parms: Mapping,
    *,
    mm_per_px: float | None = None,
    hough_localmax_fallback: bool = False,
    hough_fallback_min_mass_below_10deg: float = 0.25,
    hough_fallback_min_gap_to_lower_deg: float = 4.0,
) -> dict[str, Any]:
    """Select the signed fascicle angle range with the best stable seed cluster."""

    scored: list[dict[str, Any]] = []
    for angle_min, angle_max in candidate_signed_fascicle_angle_ranges(parms):
        try:
            result = score_fascicle_angle_range(
                seed_frames,
                parms,
                angle_min_deg=angle_min,
                angle_max_deg=angle_max,
                mm_per_px=mm_per_px,
                hough_localmax_fallback=bool(hough_localmax_fallback),
                hough_fallback_min_mass_below_10deg=float(hough_fallback_min_mass_below_10deg),
                hough_fallback_min_gap_to_lower_deg=float(hough_fallback_min_gap_to_lower_deg),
            )
        except Exception as exc:  # pragma: no cover - diagnostic path for unexpected detector failures.
            result = {
                "angle_min_deg": float(angle_min),
                "angle_max_deg": float(angle_max),
                "score": -np.inf,
                "top_cluster": {},
                "parms": None,
                "seed_entries": None,
                "seed_config": None,
                "candidates": None,
                "clusters": None,
                "error": str(exc),
            }
        scored.append(result)

    best = select_best_fascicle_angle_range_result(scored)
    return {
        "selected": bool(best and np.isfinite(float(best.get("score", -np.inf)))),
        "best": best,
        "candidates": scored,
    }


def select_best_fascicle_angle_range_result(
    scored: Sequence[Mapping[str, Any]],
    *,
    positive_tie_tolerance: float = AUTO_ORIENTATION_SIGN_TIE_TOL,
) -> Mapping[str, Any]:
    """Return the best auto-orientation result, preferring positive angles in near-ties."""

    ranked = sorted(
        scored,
        key=lambda item: float(item["score"]) if np.isfinite(float(item["score"])) else -np.inf,
        reverse=True,
    )
    if not ranked:
        return {}

    best = ranked[0]
    best_score = float(best.get("score", -np.inf))
    positive = [
        item
        for item in ranked
        if float(item.get("angle_min_deg", np.nan)) > 0.0 and float(item.get("angle_max_deg", np.nan)) > 0.0
    ]
    if not positive or not np.isfinite(best_score):
        return best

    best_positive = positive[0]
    positive_score = float(best_positive.get("score", -np.inf))
    if np.isfinite(positive_score) and positive_score >= best_score - float(positive_tie_tolerance):
        return best_positive
    return best


def fascicle_angle_auto_diagnostics(auto_result: Mapping[str, Any], selected_range: tuple[float, float]) -> list[dict[str, Any]]:
    """Build JSON-safe diagnostics for fascicle angle auto-selection."""

    rows: list[dict[str, Any]] = []
    for item in auto_result.get("candidates", []):
        score = float(item.get("score", np.nan))
        top_cluster = item.get("top_cluster", {}) or {}
        angle_min = float(item.get("angle_min_deg", np.nan))
        angle_max = float(item.get("angle_max_deg", np.nan))
        rows.append(
            {
                "angle_min_deg": angle_min,
                "angle_max_deg": angle_max,
                "selected": bool(np.isclose(angle_min, selected_range[0]) and np.isclose(angle_max, selected_range[1])),
                "score": score if np.isfinite(score) else None,
                "cluster_id": str(top_cluster.get("cluster_id", "")),
                "frame_coverage": int(top_cluster.get("frame_coverage", 0) or 0),
                "median_alpha_deg": (
                    float(top_cluster["median_alpha_deg"])
                    if "median_alpha_deg" in top_cluster and np.isfinite(float(top_cluster["median_alpha_deg"]))
                    else None
                ),
                "error": item.get("error"),
            }
        )
    return rows


def geofeature_measurement_lines(geofeatures: list[Mapping], width: int) -> tuple[np.ndarray, np.ndarray]:
    super_lines = []
    deep_lines = []
    for entry in geofeatures:
        super_pos = np.asarray(entry["super_pos"], dtype=np.float64).reshape(-1)
        deep_pos = np.asarray(entry["deep_pos"], dtype=np.float64).reshape(-1)
        super_lines.append([1.0, super_pos[0], float(width), super_pos[1]])
        deep_lines.append([1.0, deep_pos[0], float(width), deep_pos[1]])
    return np.asarray(super_lines, dtype=np.float64), np.asarray(deep_lines, dtype=np.float64)


def geofeature_alpha(geofeatures: list[Mapping]) -> np.ndarray:
    return np.asarray([float(np.asarray(entry["alpha"]).reshape(-1)[0]) for entry in geofeatures], dtype=np.float64)


def _entry_candidate_arrays(entry: Mapping, *, max_candidates: int = 10) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alphas = np.asarray(entry.get("alphas", []), dtype=np.float64).reshape(-1)[:max_candidates]
    weights = np.asarray(entry.get("weights", entry.get("ws", [])), dtype=np.float64).reshape(-1)[:max_candidates]
    x_lines = np.asarray(entry.get("x", []), dtype=np.float64)
    y_lines = np.asarray(entry.get("y", []), dtype=np.float64)
    if x_lines.ndim != 2 or x_lines.shape[1] < 2:
        x_lines = np.full((len(alphas), 2), np.nan, dtype=np.float64)
    else:
        x_lines = x_lines[:max_candidates, :2]
    if y_lines.ndim != 2 or y_lines.shape[1] < 2:
        y_lines = np.full((len(alphas), 2), np.nan, dtype=np.float64)
    else:
        y_lines = y_lines[:max_candidates, :2]
    n = min(len(alphas), len(weights), len(x_lines), len(y_lines))
    return alphas[:n], weights[:n], x_lines[:n], y_lines[:n]


def select_fascicle_candidate_persistence(
    geofeatures: Sequence[Mapping],
    raw_alpha_deg: np.ndarray,
    *,
    config: FascicleCandidatePersistenceConfig,
) -> dict[str, Any]:
    """
    Prefer fascicle Hough candidates that persist from the previous selected angle.

    When disabled, this still returns candidate rows for debug output while
    keeping the raw TimTrack alpha unchanged.
    """

    raw_alpha = np.asarray(raw_alpha_deg, dtype=np.float64).reshape(-1)
    n = min(len(geofeatures), len(raw_alpha))
    selected = raw_alpha[:n].copy()
    selected_idx = np.full(n, -1, dtype=np.int32)
    raw_rejected = np.zeros(n, dtype=bool)
    reasons = np.full(n, "raw TimTrack alpha kept", dtype=object)
    candidate_rows: list[dict[str, Any]] = []
    selection_rows: list[dict[str, Any]] = []

    for frame in range(n):
        entry = geofeatures[frame]
        alphas, weights, x_lines, y_lines = _entry_candidate_arrays(entry)
        finite_weight = np.isfinite(weights) & (weights > 0)
        max_weight = float(np.nanmax(weights[finite_weight])) if np.any(finite_weight) else np.nan
        valid = (
            np.isfinite(alphas)
            & (alphas >= float(config.angle_min_deg))
            & (alphas <= float(config.angle_max_deg))
        )

        nearest_raw_idx = -1
        if len(alphas) and np.isfinite(raw_alpha[frame]):
            finite_alpha = np.isfinite(alphas)
            if np.any(finite_alpha):
                nearest_raw_idx = int(np.flatnonzero(finite_alpha)[np.nanargmin(np.abs(alphas[finite_alpha] - raw_alpha[frame]))])

        if frame == 0 or not bool(config.enabled) or not np.isfinite(raw_alpha[frame]):
            selected[frame] = raw_alpha[frame]
            selected_idx[frame] = nearest_raw_idx
            reasons[frame] = "candidate persistence disabled" if not bool(config.enabled) else "first frame"
        else:
            previous = selected[frame - 1]
            raw_jump = abs(float(raw_alpha[frame] - previous)) if np.isfinite(previous) else 0.0
            if raw_jump <= float(config.max_angle_step_deg) or not np.any(valid):
                selected[frame] = raw_alpha[frame]
                selected_idx[frame] = nearest_raw_idx
                if not np.any(valid):
                    reasons[frame] = "no valid candidate in configured angle range"
                else:
                    reasons[frame] = f"raw alpha jump {raw_jump:.2f}deg within limit"
            else:
                candidate_indices = np.flatnonzero(valid)
                candidate_alphas = alphas[candidate_indices]
                candidate_weights = weights[candidate_indices]
                if np.isfinite(max_weight) and max_weight > 0:
                    weight_norm = np.nan_to_num(candidate_weights / max_weight, nan=0.0)
                else:
                    weight_norm = np.zeros_like(candidate_alphas)
                costs = np.abs(candidate_alphas - previous) - float(config.hough_weight_bonus_deg) * weight_norm
                best_pos = int(np.nanargmin(costs))
                best_idx = int(candidate_indices[best_pos])
                best_alpha = float(alphas[best_idx])
                best_jump = abs(best_alpha - previous)
                if best_jump <= float(config.max_angle_step_deg):
                    selected[frame] = best_alpha
                    selected_idx[frame] = best_idx
                    raw_rejected[frame] = abs(best_alpha - raw_alpha[frame]) > 1e-9
                    reasons[frame] = (
                        f"raw alpha jump {raw_jump:.2f}deg; selected candidate {best_alpha:.2f}deg "
                        f"near previous {previous:.2f}deg"
                    )
                else:
                    selected[frame] = raw_alpha[frame]
                    selected_idx[frame] = nearest_raw_idx
                    reasons[frame] = (
                        f"raw alpha jump {raw_jump:.2f}deg; no candidate within "
                        f"{config.max_angle_step_deg:.2f}deg of previous"
                    )

        selection_rows.append(
            {
                "Frame": int(frame),
                "raw_alpha_deg": float(raw_alpha[frame]),
                "selected_alpha_deg": float(selected[frame]),
                "selected_candidate_idx": int(selected_idx[frame]),
                "raw_alpha_rejected": bool(raw_rejected[frame]),
                "reason": str(reasons[frame]),
                "n_candidates": int(len(alphas)),
                "angle_min_deg": float(config.angle_min_deg),
                "angle_max_deg": float(config.angle_max_deg),
                "max_angle_step_deg": float(config.max_angle_step_deg),
            }
        )

        for cand_idx, (alpha, weight) in enumerate(zip(alphas, weights)):
            is_selected = cand_idx == int(selected_idx[frame])
            weight_norm = (
                float(weight / max_weight)
                if np.isfinite(weight) and np.isfinite(max_weight) and max_weight > 0
                else np.nan
            )
            candidate_rows.append(
                {
                    "Frame": int(frame),
                    "candidate_idx": int(cand_idx),
                    "candidate_alpha_deg": float(alpha),
                    "candidate_weight": float(weight),
                    "candidate_weight_norm": weight_norm,
                    "x1": float(x_lines[cand_idx, 0]) if cand_idx < len(x_lines) else np.nan,
                    "y1": float(y_lines[cand_idx, 0]) if cand_idx < len(y_lines) else np.nan,
                    "x2": float(x_lines[cand_idx, 1]) if cand_idx < len(x_lines) else np.nan,
                    "y2": float(y_lines[cand_idx, 1]) if cand_idx < len(y_lines) else np.nan,
                    "inside_configured_angle_range": bool(valid[cand_idx]) if cand_idx < len(valid) else False,
                    "selected": bool(is_selected),
                    "selection_reason": str(reasons[frame]) if is_selected else "not selected",
                }
            )

    return {
        "selected_alpha_deg": selected,
        "selected_candidate_idx": selected_idx,
        "raw_alpha_rejected": raw_rejected,
        "selection_reason": reasons.astype(str),
        "candidate_rows": candidate_rows,
        "selection_rows": selection_rows,
    }


def _line_mid_angle(line_1b: np.ndarray) -> tuple[float, float]:
    line = np.asarray(line_1b, dtype=np.float64).reshape(4)
    dx = line[2] - line[0]
    slope = (line[3] - line[1]) / dx if abs(dx) > 1e-12 else np.nan
    angle = -float(np.rad2deg(np.arctan2(slope, 1.0))) if np.isfinite(slope) else np.nan
    return float((line[1] + line[3]) / 2.0), angle


def aponeurosis_gating_rows(apo_state: Mapping[str, np.ndarray], super_meas: np.ndarray, deep_meas: np.ndarray) -> list[dict[str, Any]]:
    """Return per-frame aponeurosis gating diagnostics for CSV output."""

    rows: list[dict[str, Any]] = []
    n = min(len(apo_state["super_lines"]), len(super_meas), len(deep_meas))
    prior_super = np.asarray(apo_state.get("prior_super_lines", apo_state["super_lines"]), dtype=np.float64)
    prior_deep = np.asarray(apo_state.get("prior_deep_lines", apo_state["deep_lines"]), dtype=np.float64)
    rejected = np.asarray(apo_state.get("line_rejected", np.zeros((n, 2), dtype=bool)), dtype=bool)
    soft = np.asarray(apo_state.get("line_soft_downweighted", np.zeros((n, 2), dtype=bool)), dtype=bool)
    r_scale = np.asarray(apo_state.get("gating_r_scale", np.ones((n, 4), dtype=np.float64)), dtype=np.float64)
    reasons = np.asarray(apo_state.get("gating_reasons", np.full((n, 2), "", dtype=object)), dtype=object)
    consecutive = np.asarray(apo_state.get("consecutive_rejections", np.zeros((n, 2), dtype=np.int32)), dtype=np.int32)

    for frame in range(n):
        sup_meas_mid, sup_meas_angle = _line_mid_angle(super_meas[frame])
        deep_meas_mid, deep_meas_angle = _line_mid_angle(deep_meas[frame])
        sup_filtered_mid, sup_filtered_angle = _line_mid_angle(apo_state["super_lines"][frame])
        deep_filtered_mid, deep_filtered_angle = _line_mid_angle(apo_state["deep_lines"][frame])
        sup_prior_mid, sup_prior_angle = _line_mid_angle(prior_super[frame])
        deep_prior_mid, deep_prior_angle = _line_mid_angle(prior_deep[frame])
        rows.append(
            {
                "Frame": int(frame),
                "super_measurement_mid_y": sup_meas_mid,
                "super_measurement_angle_deg": sup_meas_angle,
                "super_prior_mid_y": sup_prior_mid,
                "super_prior_angle_deg": sup_prior_angle,
                "super_filtered_mid_y": sup_filtered_mid,
                "super_filtered_angle_deg": sup_filtered_angle,
                "super_line_rejected": bool(rejected[frame, 0]),
                "super_line_soft_downweighted": bool(soft[frame, 0]),
                "super_endpoint_r_scale_left": float(r_scale[frame, 0]),
                "super_endpoint_r_scale_right": float(r_scale[frame, 1]),
                "super_consecutive_rejections": int(consecutive[frame, 0]),
                "super_reason": str(reasons[frame, 0]),
                "deep_measurement_mid_y": deep_meas_mid,
                "deep_measurement_angle_deg": deep_meas_angle,
                "deep_prior_mid_y": deep_prior_mid,
                "deep_prior_angle_deg": deep_prior_angle,
                "deep_filtered_mid_y": deep_filtered_mid,
                "deep_filtered_angle_deg": deep_filtered_angle,
                "deep_line_rejected": bool(rejected[frame, 1]),
                "deep_line_soft_downweighted": bool(soft[frame, 1]),
                "deep_endpoint_r_scale_left": float(r_scale[frame, 2]),
                "deep_endpoint_r_scale_right": float(r_scale[frame, 3]),
                "deep_consecutive_rejections": int(consecutive[frame, 1]),
                "deep_reason": str(reasons[frame, 1]),
            }
        )
    return rows


def confidence_debug_rows(confidence_arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Return confidence/adaptive-R terms as CSV rows."""

    if not confidence_arrays:
        return []
    n = len(next(iter(confidence_arrays.values())))
    rows: list[dict[str, Any]] = []
    for frame in range(n):
        row: dict[str, Any] = {"Frame": int(frame)}
        for key, values in confidence_arrays.items():
            arr = np.asarray(values)
            if arr.ndim != 1 or len(arr) <= frame:
                continue
            value = arr[frame]
            row[key] = bool(value) if arr.dtype == np.dtype(bool) else float(value)
        rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def print_time_series_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    print("\nANG/PEN/FL over time:")
    print("Frame,Time_s,ANG_deg,PEN_deg,FL")
    for row in rows:
        print(
            f"{int(row['Frame'])},"
            f"{float(row['Time']):.6f},"
            f"{float(row['ANG']):.6f},"
            f"{float(row['PEN']):.6f},"
            f"{float(row['FL']):.6f}"
        )


def draw_line_1b(image: np.ndarray, segment_1b: np.ndarray, color: tuple[int, int, int], thickness: int) -> None:
    if segment_1b is None or not np.all(np.isfinite(segment_1b)):
        return
    line = np.asarray(segment_1b, dtype=np.float64).reshape(4).copy()
    line -= np.asarray([1.0, 1.0, 1.0, 1.0])
    ut.draw_line_on_image(image, line, color=color, thickness=thickness)


def draw_overlay_frame(
    frame: np.ndarray,
    rois: Mapping[str, roi.ROI],
    arrays: Mapping[str, np.ndarray],
    result_idx: int,
    *,
    show_kalman_comparison: bool = False,
) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
    vis = cv2.cvtColor(roi.ensure_uint8_image(gray), cv2.COLOR_GRAY2BGR)
    roi_colors = {
        "superficial": (255, 0, 0),
        "deep": (0, 255, 0),
        "fascicle": (0, 255, 255),
    }
    for name, box in rois.items():
        x, y, w, h = map(int, box)
        cv2.rectangle(vis, (x, y), (x + w, y + h), roi_colors.get(name, (255, 255, 255)), 1)
    draw_line_1b(vis, arrays["sup_apo_lines"][result_idx], (255, 0, 0), 2)
    draw_line_1b(vis, arrays["deep_apo_lines"][result_idx], (0, 255, 0), 2)
    draw_line_1b(vis, arrays["klt_prior_segments"][result_idx], (0, 165, 255), 1)
    if show_kalman_comparison and "fixed_fascicle_segments" in arrays:
        draw_line_1b(vis, arrays["fixed_fascicle_segments"][result_idx], (0, 0, 255), 5)
        draw_line_1b(vis, arrays["fascicle_segments"][result_idx], (255, 0, 0), 3)
    else:
        draw_line_1b(vis, arrays["fascicle_segments"][result_idx], (0, 0, 255), 3)

    fl = arrays["FL_mm"][result_idx] if "FL_mm" in arrays else np.nan
    text_lines = [
        f"Frame: {int(arrays['frame'][result_idx])}",
        f"ANG: {arrays['ANG_deg'][result_idx]:.2f} deg",
        f"PEN: {arrays['PEN_deg'][result_idx]:.2f} deg",
        f"FL: {fl:.2f} mm" if np.isfinite(fl) else f"FL: {arrays['FL_px'][result_idx]:.1f} px",
    ]
    if "combined_confidence" in arrays:
        conf = float(arrays["combined_confidence"][result_idx])
        scale = float(arrays.get("r_scale", np.full_like(arrays["combined_confidence"], np.nan))[result_idx])
        text_lines.append(f"Conf: {conf:.2f}  R x{scale:.1f}")
    if show_kalman_comparison and "fixed_fascicle_segments" in arrays:
        text_lines.append("Normal Kalman: red")
        text_lines.append("Adaptive Kalman: blue")
    ut.put_text_lines_on_image(
        vis,
        text_lines,
        origin=(30, 35),
        line_spacing=24,
        font_scale=0.65,
        color=(255, 255, 255),
        outline_color=(0, 0, 0),
    )
    if "combined_confidence" in arrays:
        cv2.circle(vis, (18, 35 + 24 * (len(text_lines) - 1) - 7), 7, _confidence_color_bgr(conf), -1)
    return vis


def save_annotated_video(
    video_path: Path,
    output_path: Path,
    rois: Mapping[str, roi.ROI],
    arrays: Mapping[str, np.ndarray],
    fps: float,
    *,
    show_kalman_comparison: bool = False,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    frame_indices = np.asarray(arrays["frame"], dtype=int)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_indices[0]))
    ok, first = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError("Could not read first frame for annotated video.")
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps if fps and fps > 0 else 30.0,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Could not create video writer: {output_path}")

    for result_idx, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
        ok, frame = cap.read()
        if not ok:
            continue
        vis = draw_overlay_frame(
            frame,
            rois,
            arrays,
            result_idx,
            show_kalman_comparison=show_kalman_comparison,
        )
        if vis.shape[:2] != (height, width):
            vis = cv2.resize(vis, (width, height))
        writer.write(vis)
        if result_idx and result_idx % 250 == 0:
            print(f"Annotated video frame {result_idx}/{len(frame_indices)}")
    writer.release()
    cap.release()
    print(f"Annotated video: {output_path}")


def save_overlay_images(
    video_path: Path,
    output_dir: Path,
    name: str,
    rois: Mapping[str, roi.ROI],
    arrays: Mapping[str, np.ndarray],
    count: int,
) -> list[Path]:
    if count <= 0:
        return []
    output_dir.mkdir(parents=True, exist_ok=True)
    frame_indices = np.asarray(arrays["frame"], dtype=int)
    positions = np.linspace(0, len(frame_indices) - 1, min(int(count), len(frame_indices)))
    selected = [int(round(pos)) for pos in positions]
    cap = cv2.VideoCapture(str(video_path))
    paths: list[Path] = []
    for result_idx in selected:
        frame_idx = int(frame_indices[result_idx])
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            continue
        vis = draw_overlay_frame(frame, rois, arrays, result_idx)
        path = output_dir / f"{name}_frame_{frame_idx:06d}.png"
        cv2.imwrite(str(path), vis)
        paths.append(path)
    cap.release()
    return paths


def save_time_series_plot(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    ang = np.asarray(arrays["ANG_deg"], dtype=np.float64)
    pen = np.asarray(arrays["PEN_deg"], dtype=np.float64)
    if "FL_mm" in arrays:
        fl = np.asarray(arrays["FL_mm"], dtype=np.float64)
        fl_label = "FL (mm)"
        fl_key = "FL_mm"
    else:
        fl = np.asarray(arrays["FL_px"], dtype=np.float64)
        fl_label = "FL (px)"
        fl_key = "FL_px"

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(11, 7), sharex=True)
    for ax, values, key, label, color in [
        (axes[0], ang, "ANG_deg", "ANG (deg)", "tab:red"),
        (axes[1], pen, "PEN_deg", "PEN (deg)", "tab:blue"),
        (axes[2], fl, fl_key, fl_label, "tab:green"),
    ]:
        fixed_key = f"fixed_{key}"
        if fixed_key in arrays:
            ax.plot(
                time_s,
                np.asarray(arrays[fixed_key], dtype=np.float64),
                color="black",
                linewidth=1.0,
                label="normal Kalman",
            )
            ax.plot(time_s, values, color=color, linewidth=1.2, label="selected Kalman")
            ax.legend(loc="best", fontsize=8)
        else:
            ax.plot(time_s, values, color=color, linewidth=1.5)
        ax.set_ylabel(label)
        ax.grid(True, alpha=0.25)
    axes[-1].set_xlabel("Time (s)")
    fig.suptitle("Strict Python UltraTimTrack outputs over time")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def run_fascicle_kalman_mode(
    mode: str,
    klt_segments: np.ndarray,
    timtrack_alpha: np.ndarray,
    super_lines: np.ndarray,
    deep_lines: np.ndarray,
    kalman_config: MatlabTwoStateKalmanConfig,
    *,
    prediction_affine_matrices: np.ndarray | None = None,
    confidence_arrays: Mapping[str, np.ndarray] | None = None,
    mm_per_pixel: float | None = None,
) -> dict[str, np.ndarray]:
    """Run the strict 2-state Kalman branch in fixed, scalar, or anisotropic mode."""

    if mode not in KALMAN_MODES:
        raise ValueError(f"Unsupported Kalman mode {mode!r}; expected one of {KALMAN_MODES}.")

    confidence_arrays = confidence_arrays or {}
    use_adaptive = kalman_mode_uses_confidence(mode)
    if use_adaptive and "r_scale" not in confidence_arrays:
        raise ValueError(f"{mode} requires confidence_arrays with an 'r_scale' entry.")

    kalman_run_config = replace(kalman_config, use_adaptive_R=use_adaptive)
    kwargs: dict[str, np.ndarray | None] = {}
    if mode == "adaptive-scalar":
        kwargs["measurement_r_scale"] = confidence_arrays.get("r_scale")
    elif mode == "adaptive-anisotropic":
        kwargs["measurement_r_scale"] = confidence_arrays.get("r_scale")
        kwargs["measurement_r_scale_theta"] = confidence_arrays.get("r_scale_theta")
        kwargs["measurement_r_scale_length"] = confidence_arrays.get("r_scale_length")

    return run_matlab_2state_kalman(
        klt_segments,
        timtrack_alpha,
        super_lines,
        deep_lines,
        config=kalman_run_config,
        mm_per_pixel=mm_per_pixel,
        prediction_affine_matrices=prediction_affine_matrices,
        **kwargs,
    )


def add_kalman_state_arrays(
    arrays: dict[str, np.ndarray],
    result: Mapping[str, np.ndarray],
    *,
    prefix: str = "",
) -> None:
    """Save parity-friendly internal Kalman arrays into the output bundle."""

    def save(key: str, source_key: str | None = None) -> None:
        value = result.get(source_key or key)
        if value is not None:
            arrays[f"{prefix}{key}"] = np.asarray(value)

    for key in [
        "X_plus",
        "X_smooth",
        "X_minus",
        "fas_p",
        "fas_p_smooth",
        "fas_p_minus",
        "forward_X_plus",
        "forward_fas_p",
        "kalman_gain",
        "smoother_gain",
        "measurement_R_diag",
        "measurement_r_scale",
        "measurement_r_scale_theta",
        "measurement_r_scale_length",
        "predicted_segments",
        "previous_corrected_segments",
        "prediction_used_affine",
        "forward_fascicle_segments",
        "forward_fascicle_end_segments",
        "forward_ANG_deg",
        "forward_PEN_deg",
        "forward_FL_px",
        "forward_FL_mm",
    ]:
        save(key)


def _finite_delta_stats(estimate: np.ndarray, reference: np.ndarray) -> dict[str, float | int]:
    estimate_arr = np.asarray(estimate, dtype=np.float64).reshape(-1)
    reference_arr = np.asarray(reference, dtype=np.float64).reshape(-1)
    n = min(len(estimate_arr), len(reference_arr))
    delta = estimate_arr[:n] - reference_arr[:n]
    valid = np.isfinite(delta)
    if not np.any(valid):
        return {
            "n_valid": 0,
            "mean_delta": np.nan,
            "mean_abs_delta": np.nan,
            "median_abs_delta": np.nan,
            "rmse_delta": np.nan,
            "max_abs_delta": np.nan,
        }
    d = delta[valid]
    abs_d = np.abs(d)
    return {
        "n_valid": int(len(d)),
        "mean_delta": float(np.mean(d)),
        "mean_abs_delta": float(np.mean(abs_d)),
        "median_abs_delta": float(np.median(abs_d)),
        "rmse_delta": float(np.sqrt(np.mean(d * d))),
        "max_abs_delta": float(np.max(abs_d)),
    }


def kalman_comparison_rows(
    selected: Mapping[str, np.ndarray],
    fixed: Mapping[str, np.ndarray],
) -> list[dict[str, float | int | str]]:
    """Summarize selected Kalman outputs minus the normal fixed-R Kalman outputs."""

    rows: list[dict[str, float | int | str]] = []
    metric_specs = [
        ("ANG", "ANG_deg", "deg"),
        ("PEN", "PEN_deg", "deg"),
        (
            "FL",
            "FL_mm" if "FL_mm" in selected and "FL_mm" in fixed else "FL_px",
            "mm" if "FL_mm" in selected and "FL_mm" in fixed else "px",
        ),
    ]
    for metric, key, unit in metric_specs:
        stats = _finite_delta_stats(selected[key], fixed[key])
        rows.append({"metric": metric, "key": key, "unit": unit, **stats})
    return rows


def _segment_length_px(segment: np.ndarray) -> float:
    line = np.asarray(segment, dtype=np.float64).reshape(-1)
    if line.size < 4 or not np.all(np.isfinite(line[:4])):
        return float("nan")
    return float(np.hypot(line[2] - line[0], line[3] - line[1]))


def _confidence_color_bgr(confidence: float) -> tuple[int, int, int]:
    if not np.isfinite(confidence):
        return (180, 180, 180)
    if confidence >= 0.75:
        return (0, 180, 0)
    if confidence >= 0.45:
        return (0, 165, 255)
    return (0, 0, 220)


def compute_confidence_series(
    frames: list[np.ndarray],
    rois: Mapping[str, roi.ROI],
    geofeatures: list[Mapping],
    fascicle_prior: np.ndarray,
    timtrack_alpha: np.ndarray,
    *,
    config: SpeckleConfidenceConfig | None = None,
    progress_every: int | None = None,
) -> dict[str, np.ndarray]:
    """Compute per-frame confidence metrics used by adaptive-R mode."""

    cfg = config or SpeckleConfidenceConfig()
    n = min(len(frames), len(geofeatures), len(fascicle_prior), len(timtrack_alpha))
    out: dict[str, np.ndarray] = {
        "speckle_zncc": np.full(n, np.nan, dtype=np.float64),
        "speckle_confidence": np.ones(n, dtype=np.float64),
        "forward_backward_error": np.full(n, np.nan, dtype=np.float64),
        "valid_patch_fraction": np.full(n, np.nan, dtype=np.float64),
        "n_valid_patches": np.zeros(n, dtype=np.int32),
        "n_total_patches": np.zeros(n, dtype=np.int32),
        "motion_consistency": np.ones(n, dtype=np.float64),
        "motion_spread_px": np.full(n, np.nan, dtype=np.float64),
        "n_motion_points": np.zeros(n, dtype=np.int32),
        "feature_reliability": np.ones(n, dtype=np.float64),
        "feature_peak_score": np.full(n, np.nan, dtype=np.float64),
        "feature_peak_count_score": np.full(n, np.nan, dtype=np.float64),
        "feature_mask_score": np.full(n, np.nan, dtype=np.float64),
        "feature_mask_density": np.full(n, np.nan, dtype=np.float64),
        "geometry_stability": np.ones(n, dtype=np.float64),
        "geometry_alpha_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_pennation_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_length_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_angle_jump_deg": np.full(n, np.nan, dtype=np.float64),
        "geometry_angle_jump_score": np.full(n, np.nan, dtype=np.float64),
        "geometry_length_jump_px": np.full(n, np.nan, dtype=np.float64),
        "geometry_length_jump_score": np.full(n, np.nan, dtype=np.float64),
        "confidence_theta": np.ones(n, dtype=np.float64),
        "confidence_length": np.ones(n, dtype=np.float64),
        "combined_confidence": np.ones(n, dtype=np.float64),
        "r_scale": np.ones(n, dtype=np.float64),
        "r_scale_theta": np.ones(n, dtype=np.float64),
        "r_scale_length": np.ones(n, dtype=np.float64),
        "detection_success": np.ones(n, dtype=bool),
    }

    fascicle_roi = rois.get("fascicle")
    prior_lengths = np.asarray([_segment_length_px(seg) for seg in np.asarray(fascicle_prior)[:n]], dtype=np.float64)

    for idx in range(n):
        entry = geofeatures[idx]
        feature = compute_feature_detection_reliability(entry, config=cfg)
        geometry = compute_geometry_stability(
            alpha_deg=float(timtrack_alpha[idx]) if np.isfinite(timtrack_alpha[idx]) else None,
            fascicle_length_px=float(prior_lengths[idx]) if np.isfinite(prior_lengths[idx]) else None,
            segment=np.asarray(fascicle_prior[idx], dtype=np.float64),
            previous_alpha_deg=float(timtrack_alpha[idx - 1]) if idx > 0 and np.isfinite(timtrack_alpha[idx - 1]) else None,
            previous_length_px=float(prior_lengths[idx - 1]) if idx > 0 and np.isfinite(prior_lengths[idx - 1]) else None,
            previous_segment=np.asarray(fascicle_prior[idx - 1], dtype=np.float64) if idx > 0 else None,
            config=cfg,
        )

        if idx > 0:
            speckle = compute_speckle_coherence(frames[idx - 1], frames[idx], roi=fascicle_roi, config=cfg)
            motion = compute_motion_consistency(
                speckle["points_prev"],
                speckle["points_curr"],
                roi=fascicle_roi,
                config=cfg,
            )
            out["speckle_zncc"][idx] = float(speckle["speckle_zncc"])
            out["speckle_confidence"][idx] = float(speckle["speckle_confidence"])
            out["forward_backward_error"][idx] = float(speckle["forward_backward_error"])
            out["valid_patch_fraction"][idx] = float(speckle["valid_patch_fraction"])
            out["n_valid_patches"][idx] = int(speckle["n_valid_patches"])
            out["n_total_patches"][idx] = int(speckle["n_total_patches"])
            out["motion_consistency"][idx] = float(motion["motion_consistency"])
            out["motion_spread_px"][idx] = float(motion["motion_spread_px"])
            out["n_motion_points"][idx] = int(motion["n_motion_points"])

        out["feature_reliability"][idx] = float(feature["feature_reliability"])
        out["feature_peak_score"][idx] = float(feature["feature_peak_score"])
        out["feature_peak_count_score"][idx] = float(feature["feature_peak_count_score"])
        out["feature_mask_score"][idx] = float(feature["feature_mask_score"])
        out["feature_mask_density"][idx] = float(feature.get("feature_mask_density", np.nan))
        out["detection_success"][idx] = bool(feature["detection_success"])
        out["geometry_stability"][idx] = float(geometry["geometry_stability"])
        out["geometry_alpha_score"][idx] = float(geometry["geometry_alpha_score"])
        out["geometry_pennation_score"][idx] = float(geometry["geometry_pennation_score"])
        out["geometry_length_score"][idx] = float(geometry["geometry_length_score"])
        out["geometry_angle_jump_deg"][idx] = float(geometry["geometry_angle_jump_deg"])
        out["geometry_angle_jump_score"][idx] = float(geometry["geometry_angle_jump_score"])
        out["geometry_length_jump_px"][idx] = float(geometry["geometry_length_jump_px"])
        out["geometry_length_jump_score"][idx] = float(geometry["geometry_length_jump_score"])
        combined = combine_confidence_metrics(
            {
                "speckle_confidence": out["speckle_confidence"][idx],
                "motion_consistency": out["motion_consistency"][idx],
                "feature_reliability": out["feature_reliability"][idx],
                "geometry_stability": out["geometry_stability"][idx],
            },
            config=cfg,
        )
        out["combined_confidence"][idx] = combined
        out["r_scale"][idx] = confidence_to_r_scale(combined, cfg)
        anisotropic_conf = combine_anisotropic_confidence_metrics(
            {
                "speckle_confidence": out["speckle_confidence"][idx],
                "motion_consistency": out["motion_consistency"][idx],
                "feature_reliability": out["feature_reliability"][idx],
                "geometry_stability": out["geometry_stability"][idx],
                "geometry_alpha_score": out["geometry_alpha_score"][idx],
                "geometry_angle_jump_score": out["geometry_angle_jump_score"][idx],
                "geometry_length_score": out["geometry_length_score"][idx],
                "geometry_length_jump_score": out["geometry_length_jump_score"][idx],
            },
            config=cfg,
        )
        out["confidence_theta"][idx] = anisotropic_conf["confidence_theta"]
        out["confidence_length"][idx] = anisotropic_conf["confidence_length"]
        anisotropic_scales = anisotropic_confidence_to_r_scales(
            anisotropic_conf["confidence_theta"],
            anisotropic_conf["confidence_length"],
            cfg,
        )
        out["r_scale_theta"][idx] = anisotropic_scales["r_scale_theta"]
        out["r_scale_length"][idx] = anisotropic_scales["r_scale_length"]

        if progress_every and idx > 0 and (idx % int(progress_every) == 0 or idx == n - 1):
            print(f"confidence processed {idx + 1}/{n}")

    return out


def save_confidence_plot(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    if "combined_confidence" not in arrays:
        return
    time_s = np.asarray(arrays["time_s"], dtype=np.float64)
    confidence = np.asarray(arrays["combined_confidence"], dtype=np.float64)
    r_scale = np.asarray(arrays.get("r_scale", np.full_like(confidence, np.nan)), dtype=np.float64)
    confidence_theta = np.asarray(arrays.get("confidence_theta", confidence), dtype=np.float64)
    confidence_length = np.asarray(arrays.get("confidence_length", confidence), dtype=np.float64)
    angle = np.asarray(arrays.get("ANG_deg", np.full_like(confidence, np.nan)), dtype=np.float64)
    length = np.asarray(arrays.get("FL_mm", arrays.get("FL_px", np.full_like(confidence, np.nan))), dtype=np.float64)
    length_label = "FL (mm)" if "FL_mm" in arrays else "FL (px)"

    path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(5, 1, figsize=(11, 10.5), sharex=True)
    axes[0].plot(time_s, confidence, color="tab:purple", linewidth=1.6)
    axes[0].plot(time_s, confidence_theta, color="tab:red", linewidth=1.0, alpha=0.75, label="theta")
    axes[0].plot(time_s, confidence_length, color="tab:green", linewidth=1.0, alpha=0.75, label="length")
    axes[0].set_ylabel("confidence")
    axes[0].set_ylim(0, 1.05)
    axes[0].legend(loc="lower right", fontsize=8)
    axes[1].plot(time_s, r_scale, color="tab:orange", linewidth=1.4, label="global")
    if "r_scale_theta" in arrays:
        axes[1].plot(time_s, arrays["r_scale_theta"], color="tab:red", linewidth=1.0, alpha=0.75, label="theta")
    if "r_scale_length" in arrays:
        axes[1].plot(time_s, arrays["r_scale_length"], color="tab:green", linewidth=1.0, alpha=0.75, label="length")
    axes[1].set_ylabel("R scale")
    axes[1].legend(loc="upper right", fontsize=8)
    axes[2].plot(time_s, angle, color="tab:red", linewidth=1.2)
    axes[2].scatter(time_s, angle, c=confidence_theta, cmap="RdYlGn", vmin=0, vmax=1, s=12)
    axes[2].set_ylabel("ANG (deg)")
    axes[3].plot(time_s, length, color="tab:green", linewidth=1.2)
    axes[3].scatter(time_s, length, c=confidence_length, cmap="RdYlGn", vmin=0, vmax=1, s=12)
    axes[3].set_ylabel(length_label)
    axes[4].plot(time_s, confidence_theta - confidence_length, color="tab:gray", linewidth=1.2)
    axes[4].axhline(0.0, color="black", linewidth=0.8)
    axes[4].set_ylabel("c_theta-c_L")
    axes[4].set_xlabel("Time (s)")
    for ax in axes:
        ax.grid(True, alpha=0.25)
    fig.suptitle("Adaptive confidence diagnostics")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_arg_parser() -> argparse.ArgumentParser:
    default_export = DEFAULT_UTT_EXPORT if DEFAULT_UTT_EXPORT.exists() else LEGACY_UTT_EXPORT
    parser = argparse.ArgumentParser(description="Run strict Python UltraTimTrack and write an annotated video.")
    parser.add_argument("video", nargs="?", type=Path, default=None, help="Video path. Omit with --interactive.")
    parser.add_argument("--interactive", action="store_true", help="Choose video and ROI interactively.")
    parser.add_argument("--name", default=None, help="Output prefix. Defaults to video stem.")
    parser.add_argument("--utt-export", type=Path, default=default_export, help="UTT numeric export .mat with parameters.")
    parser.add_argument("--roi-path", type=Path, default=None, help="ROI JSON path.")
    parser.add_argument("--select-roi", action="store_true", help="Open ROI selector and save/update ROI JSON.")
    parser.add_argument("--overwrite-roi", action="store_true", help="Allow replacing an existing ROI JSON.")
    parser.add_argument(
        "--no-roi-parameter-update",
        action="store_true",
        help="Draw ROIs but do not update aponeurosis cuts or fascicle Emask from them.",
    )
    parser.add_argument(
        "--apo-maxangle",
        type=float,
        default=None,
        help="Override both aponeurosis fit angle limits in degrees, e.g. 10 allows roughly -10..+10 deg.",
    )
    parser.add_argument(
        "--super-apo-maxangle",
        type=float,
        default=None,
        help="Override only the superficial aponeurosis fit angle limit in degrees.",
    )
    parser.add_argument(
        "--deep-apo-maxangle",
        type=float,
        default=None,
        help="Override only the deep aponeurosis fit angle limit in degrees.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N frames.")
    parser.add_argument("--seed-frames", type=int, default=11, help="Frames used for autonomous seed selection.")
    parser.add_argument(
        "--seed-angle-range",
        type=float,
        nargs=2,
        metavar=("MIN_DEG", "MAX_DEG"),
        default=None,
        help=(
            "Optional seed-only fascicle angle range. This constrains autonomous initialization "
            "without changing the per-frame fascicle Hough angle range."
        ),
    )
    parser.add_argument("--fas-angle-min", type=float, default=None, help="Override minimum fascicle Hough/seed angle.")
    parser.add_argument("--fas-angle-max", type=float, default=None, help="Override maximum fascicle Hough/seed angle.")
    parser.add_argument(
        "--fas-angle-auto",
        dest="fas_angle_auto",
        action="store_true",
        default=True,
        help="Automatically choose positive or reverse-signed fascicle angle range from seed frames.",
    )
    parser.add_argument(
        "--no-fas-angle-auto",
        dest="fas_angle_auto",
        action="store_false",
        help="Disable automatic positive/reverse fascicle angle range selection.",
    )
    parser.add_argument(
        "--hough-localmax-fallback",
        action="store_true",
        help=(
            "On mass/gap flagged frames, replace MATLAB-style 2D houghpeaks with "
            "angle-profile localmax Hough output."
        ),
    )
    parser.add_argument(
        "--hough-fallback-min-mass-below-10deg",
        type=float,
        default=0.25,
        help="Minimum normalized mass below alpha-10 deg for --hough-localmax-fallback.",
    )
    parser.add_argument(
        "--hough-fallback-min-gap-to-lower-deg",
        type=float,
        default=4.0,
        help="Minimum gap to nearest lower candidate for --hough-localmax-fallback.",
    )
    parser.add_argument(
        "--candidate-persistence",
        action="store_true",
        help="Prefer fascicle candidates close to the previous selected alpha when raw alpha jumps abruptly.",
    )
    parser.add_argument(
        "--max-angle-step",
        type=float,
        default=8.0,
        help="Maximum preferred frame-to-frame fascicle alpha step for --candidate-persistence.",
    )
    parser.add_argument(
        "--candidate-weight-bonus",
        type=float,
        default=2.0,
        help="Hough-weight bonus, in degrees, used when ranking persistent fascicle candidates.",
    )
    parser.add_argument(
        "--apo-gating",
        action="store_true",
        help="Enable separate anatomical gating for superficial/deep aponeurosis measurements.",
    )
    parser.add_argument("--apo-gate-mid-innovation-px", type=float, default=10.0)
    parser.add_argument("--apo-gate-super-mid-jump-px", type=float, default=12.0)
    parser.add_argument("--apo-gate-deep-mid-jump-px", type=float, default=6.0)
    parser.add_argument("--apo-gate-angle-jump-deg", type=float, default=2.5)
    parser.add_argument("--apo-gate-max-rejections", type=int, default=3)
    parser.add_argument(
        "--debug-detections",
        action="store_true",
        help="Save fascicle candidate, selection, aponeurosis rejection, and confidence debug CSVs.",
    )
    parser.add_argument("--results-dir", type=Path, default=PROJECT_ROOT / "results" / "strict_ultratimtrack_runs")
    parser.add_argument("--annotated-video", type=Path, default=None, help="Output MP4 path.")
    parser.add_argument("--no-annotated-video", action="store_true", help="Do not write annotated MP4.")
    parser.add_argument("--save-overlays", type=int, default=3, help="Number of overlay PNGs to save.")
    parser.add_argument("--no-time-series-plot", action="store_true", help="Do not write ANG/PEN/FL time-series PNG.")
    parser.add_argument("--print-time-series", action="store_true", help="Print ANG/PEN/FL values for every processed frame.")
    parser.add_argument(
        "--kalman-mode",
        choices=KALMAN_MODES,
        default=None,
        help=(
            "Kalman measurement covariance mode: fixed normal Kalman, scalar adaptive R, "
            "or anisotropic theta/length adaptive R. Defaults to fixed unless --adaptive-r is used."
        ),
    )
    parser.add_argument(
        "--adaptive-r",
        action="store_true",
        help="Backward-compatible alias for --kalman-mode adaptive-anisotropic.",
    )
    parser.add_argument(
        "--compare-to-fixed-kalman",
        "--compare-kalman",
        dest="compare_to_fixed_kalman",
        action="store_true",
        help="Also run the normal fixed-R Kalman and save ANG/PEN/FL deltas for the selected mode.",
    )
    parser.add_argument(
        "--annotate-kalman-comparison",
        "--annotate-both-kalman",
        dest="annotate_kalman_comparison",
        action="store_true",
        help="Draw normal fixed-R Kalman in red and selected adaptive Kalman in blue on the annotated video.",
    )
    parser.add_argument(
        "--confidence-debug",
        action="store_true",
        help="Compute confidence metrics even when --kalman-mode fixed is selected.",
    )
    parser.add_argument("--save-confidence-plots", action="store_true", help="Write confidence/R-scale diagnostic PNG.")
    parser.add_argument("--mm-per-pixel", type=float, default=None, help="Override pixel scale.")
    parser.add_argument("--image-depth-mm", type=float, default=None, help="Image depth; used if --mm-per-pixel is absent.")
    parser.add_argument("--progress-every", type=int, default=250)
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    kalman_mode_from_cli = args.kalman_mode is not None or bool(args.adaptive_r)
    annotate_comparison_from_cli = bool(args.annotate_kalman_comparison)
    compare_from_cli = bool(args.compare_to_fixed_kalman or annotate_comparison_from_cli)
    if args.fas_angle_min is not None and args.fas_angle_max is not None and args.fas_angle_min >= args.fas_angle_max:
        parser.error("--fas-angle-min must be smaller than --fas-angle-max.")
    if args.seed_frames <= 0:
        parser.error("--seed-frames must be positive.")
    if args.seed_angle_range is not None and args.seed_angle_range[0] >= args.seed_angle_range[1]:
        parser.error("--seed-angle-range MIN_DEG must be smaller than MAX_DEG.")
    if args.max_angle_step <= 0:
        parser.error("--max-angle-step must be positive.")
    if args.hough_fallback_min_mass_below_10deg < 0:
        parser.error("--hough-fallback-min-mass-below-10deg must be non-negative.")
    if args.hough_fallback_min_gap_to_lower_deg < 0:
        parser.error("--hough-fallback-min-gap-to-lower-deg must be non-negative.")
    if args.apo_gate_max_rejections < 0:
        parser.error("--apo-gate-max-rejections must be non-negative.")
    if args.mm_per_pixel is not None and args.mm_per_pixel <= 0:
        parser.error("--mm-per-pixel must be positive.")
    if args.image_depth_mm is not None and args.image_depth_mm <= 0:
        parser.error("--image-depth-mm must be positive.")

    if args.kalman_mode is None:
        args.kalman_mode = "adaptive-anisotropic" if args.adaptive_r else "fixed"
    elif args.adaptive_r:
        if args.kalman_mode == "fixed":
            args.kalman_mode = "adaptive-anisotropic"
        elif args.kalman_mode != "adaptive-anisotropic":
            parser.error("--adaptive-r is an alias for --kalman-mode adaptive-anisotropic.")
    args.adaptive_r = kalman_mode_uses_confidence(args.kalman_mode)
    if args.annotate_kalman_comparison:
        if not args.adaptive_r:
            if kalman_mode_from_cli:
                parser.error("--annotate-kalman-comparison requires an adaptive Kalman mode.")
            args.kalman_mode = "adaptive-anisotropic"
            args.adaptive_r = True
        args.compare_to_fixed_kalman = True

    if args.video is None or args.interactive:
        args.video = prompt_video_from_raw(PROJECT_ROOT / "data" / "raw") if args.video is None else args.video

    args.video = (PROJECT_ROOT / args.video).resolve() if not args.video.is_absolute() else args.video.resolve()
    args.name = args.name or args.video.stem
    args.results_dir = (PROJECT_ROOT / args.results_dir).resolve() if not args.results_dir.is_absolute() else args.results_dir.resolve()
    args.utt_export = (PROJECT_ROOT / args.utt_export).resolve() if not args.utt_export.is_absolute() else args.utt_export.resolve()
    args.roi_path = args.roi_path or (PROJECT_ROOT / "data" / "rois" / f"{args.name}_rois.json")
    args.roi_path = (PROJECT_ROOT / args.roi_path).resolve() if not args.roi_path.is_absolute() else args.roi_path.resolve()
    if args.annotated_video is not None and not args.annotated_video.is_absolute():
        args.annotated_video = (PROJECT_ROOT / args.annotated_video).resolve()

    if args.interactive:
        if args.roi_path.exists():
            args.select_roi = prompt_yes_no("ROI file exists. Select new ROIs?", default=False)
            args.overwrite_roi = args.select_roi
        else:
            args.select_roi = True
            args.overwrite_roi = True
        args.no_annotated_video = not prompt_yes_no("Write annotated video?", default=True)
        if not kalman_mode_from_cli:
            args.kalman_mode = prompt_kalman_mode(default=args.kalman_mode)
        args.adaptive_r = kalman_mode_uses_confidence(args.kalman_mode)
        if args.annotate_kalman_comparison and not args.adaptive_r:
            args.annotate_kalman_comparison = False
            if not compare_from_cli:
                args.compare_to_fixed_kalman = False
        if args.adaptive_r and not compare_from_cli:
            args.compare_to_fixed_kalman = prompt_yes_no(
                "Compare adaptive Kalman to normal fixed-R Kalman?",
                default=True,
            )
        if (
            args.adaptive_r
            and args.compare_to_fixed_kalman
            and not args.no_annotated_video
            and not annotate_comparison_from_cli
        ):
            args.annotate_kalman_comparison = prompt_yes_no(
                "Draw normal and adaptive Kalman together on the annotated video?",
                default=True,
            )

    return args


def process_video(args: argparse.Namespace) -> dict[str, Path | None]:
    start_time = time.time()
    frame0, fps, n_video_frames = read_first_frame(args.video)
    n_limit = min(args.limit or n_video_frames, n_video_frames)
    if n_limit <= 1:
        raise ValueError("--limit must leave at least two frames.")

    mat_root = loadmat(args.utt_export, simplify_cells=True)["UTT_numeric_export"]
    parms = copy.deepcopy(mat_root["parms"])

    rois: dict[str, roi.ROI] = {}
    should_select_roi = args.select_roi or (not args.roi_path.exists() and not args.no_roi_parameter_update)
    if args.roi_path.exists() or should_select_roi:
        rois = select_or_load_rois(
            frame0,
            args.roi_path,
            select_roi=should_select_roi,
            overwrite_roi=args.overwrite_roi,
        )
        if not args.no_roi_parameter_update:
            parms = update_parms_from_rois(parms, rois, frame0.shape[:2])
    parms = apply_aponeurosis_maxangle_overrides(
        parms,
        apo_maxangle=args.apo_maxangle,
        super_apo_maxangle=args.super_apo_maxangle,
        deep_apo_maxangle=args.deep_apo_maxangle,
    )

    pixel_scale = resolve_measurement_pixel_scale(
        mat_root,
        image_height_px=int(frame0.shape[0]),
        mm_per_pixel=args.mm_per_pixel,
        image_depth_mm=args.image_depth_mm,
    )
    mm_per_px = float(pixel_scale["measurement_mm_per_pixel"])
    measurement_scale_source = str(pixel_scale["measurement_scale_source"])
    matlab_image_depth_mm = pixel_scale["matlab_export_image_depth_mm"]
    print(
        "\nPixel scale: "
        f"measurements={mm_per_px:.8g} mm/px ({measurement_scale_source}); "
        "fascicle seed selection uses pixel/image evidence only"
    )

    seed_frame_count = min(int(args.seed_frames), n_limit)
    seed_frames = read_gray_frames(args.video, seed_frame_count)
    print(
        f"Using {len(seed_frames)} seed frame(s) for autonomous fascicle choice "
        f"(requested {int(args.seed_frames)})."
    )
    seed_entries: list[dict[str, Any]] | None = None
    seed_config: FascicleSeedScoringConfig | None = None
    candidates = None
    clusters = None
    fas_angle_auto_result: dict[str, Any] | None = None
    fas_angle_auto_applied = False
    manual_fas_angle_override = args.fas_angle_min is not None or args.fas_angle_max is not None

    if manual_fas_angle_override:
        parms = apply_fascicle_angle_overrides(
            parms,
            fas_angle_min=args.fas_angle_min,
            fas_angle_max=args.fas_angle_max,
        )
    elif args.fas_angle_auto:
        print("\nDetecting fascicle angle orientation from seed frames...")
        fas_angle_auto_result = choose_fascicle_angle_range_from_seed_frames(
            seed_frames,
            parms,
            hough_localmax_fallback=bool(args.hough_localmax_fallback),
            hough_fallback_min_mass_below_10deg=float(args.hough_fallback_min_mass_below_10deg),
            hough_fallback_min_gap_to_lower_deg=float(args.hough_fallback_min_gap_to_lower_deg),
        )
        if fas_angle_auto_result["selected"]:
            best = fas_angle_auto_result["best"]
            parms = best["parms"]
            seed_entries = best["seed_entries"]
            seed_config = best["seed_config"]
            candidates = best["candidates"]
            clusters = best["clusters"]
            fas_angle_auto_applied = True
            top_cluster = best.get("top_cluster", {}) or {}
            cluster_id = str(top_cluster.get("cluster_id", ""))
            cluster_score = float(best["score"])
            print(
                "Auto-selected fascicle angle range: "
                f"{float(best['angle_min_deg']):.3f} to {float(best['angle_max_deg']):.3f} deg "
                f"(orientation score {cluster_score:.3f}, preflight cluster {cluster_id})"
            )
        else:
            print("Fascicle angle auto-selection found no stable seed cluster; keeping exported range.")

    fas_angle_min, fas_angle_max = current_fascicle_angle_range(parms)
    if args.seed_angle_range is None:
        seed_angle_min, seed_angle_max = fas_angle_min, fas_angle_max
    else:
        seed_angle_min, seed_angle_max = map(float, args.seed_angle_range)
        if seed_angle_min < fas_angle_min or seed_angle_max > fas_angle_max:
            raise ValueError(
                "--seed-angle-range must lie within the active per-frame fascicle angle range "
                f"[{fas_angle_min:g}, {fas_angle_max:g}] deg."
            )
        # Preflight candidates, when present, were generated for the orientation range.
        # Regenerate only the initialization candidates for the explicit seed-only range.
        seed_config = None
        candidates = None
        clusters = None
        print(
            "Using seed-only fascicle angle range: "
            f"{seed_angle_min:.3f} to {seed_angle_max:.3f} deg "
            f"(per-frame Hough remains {fas_angle_min:.3f} to {fas_angle_max:.3f} deg)"
        )

    block_size = np.asarray(mat_root.get("BlockSize", [81, 81]), dtype=int).reshape(-1)
    win_size = (int(block_size[-1]), int(block_size[0])) if block_size.size >= 2 else (81, 81)
    klt_config = UltraTrackKLTConfig(lk_win_size=win_size)
    kalman_config = MatlabTwoStateKalmanConfig(
        q_parameter=float(mat_root.get("Q", 0.01)),
        x_measurement_variance=float(mat_root.get("X", 100.0)),
        alpha_measurement_variance=float(np.asarray(mat_root.get("R", [3.05529211]), dtype=float).reshape(-1)[0]),
        n_start_frames=int(mat_root.get("NS", 1)),
        run_smoother=True,
    )
    r_values = np.asarray(mat_root.get("R", [3.05529211, 100, 100, 100, 100]), dtype=float).reshape(-1)
    apo_measurement_variance = r_values[1:5] * 0.01 if r_values.size >= 5 else np.ones(4, dtype=np.float64)

    args.results_dir.mkdir(parents=True, exist_ok=True)
    run_dir = args.results_dir / args.name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("\nRunning TimTrack image stream...")
    print(f"Using fascicle angle range: {fas_angle_min:.3f} to {fas_angle_max:.3f} deg")
    if args.hough_localmax_fallback:
        print(
            "Using conditional Hough localmax fallback: "
            f"mass_below_10deg >= {args.hough_fallback_min_mass_below_10deg:g}, "
            f"gap_to_lower_deg >= {args.hough_fallback_min_gap_to_lower_deg:g}"
        )
    geofeatures = run_timtrack_geofeatures_from_video(
        str(args.video),
        parms,
        limit=n_limit,
        keep_debug=False,
        emask_mode="matlab",
        hough_localmax_fallback=bool(args.hough_localmax_fallback),
        hough_fallback_min_mass_below_10deg=float(args.hough_fallback_min_mass_below_10deg),
        hough_fallback_min_gap_to_lower_deg=float(args.hough_fallback_min_gap_to_lower_deg),
        progress_every=args.progress_every,
    )
    n = len(geofeatures)
    if n <= 1:
        raise RuntimeError("TimTrack stream produced too few frames.")
    if len(seed_frames) > n:
        seed_frames = seed_frames[:n]
        seed_frame_count = len(seed_frames)
        if seed_entries is not None:
            seed_entries = seed_entries[:n]

    print("\nSelecting autonomous fascicle seed...")
    if seed_entries is None or seed_config is None or candidates is None or clusters is None:
        seed_entries = detect_seed_geofeature_entries(
            seed_frames,
            parms,
            hough_localmax_fallback=bool(args.hough_localmax_fallback),
            hough_fallback_min_mass_below_10deg=float(args.hough_fallback_min_mass_below_10deg),
            hough_fallback_min_gap_to_lower_deg=float(args.hough_fallback_min_gap_to_lower_deg),
        )
        seed_config = replace(
            FascicleSeedScoringConfig(min_cluster_frame_coverage=min(8, seed_frame_count)),
            angle_min_deg=seed_angle_min,
            angle_max_deg=seed_angle_max,
        )
        candidates = extract_fascicle_seed_candidates(
            seed_entries,
            seed_frames,
            config=seed_config,
        )
        candidates, clusters = cluster_seed_candidates(candidates, min_frame_coverage=seed_config.min_cluster_frame_coverage)
    else:
        print("Using preflight seed detections for selected fascicle angle range.")
    selection = select_autonomous_fascicle_seed(candidates, clusters, seed_entries[0], frame0.shape[1])
    selected_seed = np.asarray(selection["selected_seed_segment"], dtype=np.float64)
    selected_alpha = float(selection["selected_alpha_deg"])
    selected_cluster = str(selection["selected_cluster"]["cluster_id"])
    print(f"Selected seed alpha: {selected_alpha:.3f} deg ({selected_cluster})")

    candidates.to_csv(run_dir / f"{args.name}_seed_candidates.csv", index=False)
    clusters.to_csv(run_dir / f"{args.name}_seed_clusters.csv", index=False)
    selection["per_frame_best"].to_csv(run_dir / f"{args.name}_selected_seed_cluster.csv", index=False)

    print("\nEstimating fascicle KLT affines with persistent tracker state...")
    klt = run_persistent_affine_video(
        args.video,
        geofeatures,
        selected_seed,
        super_cut=np.asarray(parms["apo"]["super"]["cut"], dtype=float).reshape(-1),
        config=klt_config,
        limit=n,
        progress_every=args.progress_every,
    )
    fascicle_prior = np.asarray(klt["fascicle_segments"], dtype=np.float64)

    print("\nRunning aponeurosis state estimator...")
    sup_meas, deep_meas = geofeature_measurement_lines(geofeatures, frame0.shape[1])
    apo_gating_config = AponeurosisGatingConfig(
        enabled=bool(args.apo_gating),
        mid_innovation_px=float(args.apo_gate_mid_innovation_px),
        super_mid_jump_px=float(args.apo_gate_super_mid_jump_px),
        deep_mid_jump_px=float(args.apo_gate_deep_mid_jump_px),
        angle_jump_deg=float(args.apo_gate_angle_jump_deg),
        max_consecutive_rejections=int(args.apo_gate_max_rejections),
        super_maxangle_deg=float(parms["apo"]["super"].get("maxangle", np.nan)),
        deep_maxangle_deg=float(parms["apo"]["deep"].get("maxangle", np.nan)),
    )
    apo_state = run_matlab_aponeurosis_state_video(
        args.video,
        geofeatures,
        sup_meas,
        deep_meas,
        super_cut=np.asarray(parms["apo"]["super"]["cut"], dtype=float).reshape(-1),
        deep_cut=np.asarray(parms["apo"]["deep"]["cut"], dtype=float).reshape(-1),
        q_parameter=float(mat_root.get("Q", 0.01)),
        measurement_variance=apo_measurement_variance,
        config=klt_config,
        gating_config=apo_gating_config,
        limit=n,
        progress_every=args.progress_every,
    )

    timtrack_alpha_raw = geofeature_alpha(geofeatures)
    hough_fallback_used = np.asarray(
        [bool(entry.get("hough_localmax_fallback_used", False)) for entry in geofeatures],
        dtype=bool,
    )
    hough_fallback_mass = np.asarray(
        [float(entry.get("hough_localmax_fallback_mass_below_10deg", np.nan)) for entry in geofeatures],
        dtype=np.float64,
    )
    hough_fallback_gap = np.asarray(
        [float(entry.get("hough_localmax_fallback_gap_to_lower_deg", np.nan)) for entry in geofeatures],
        dtype=np.float64,
    )
    hough_baseline_alpha = np.asarray(
        [float(entry.get("hough_baseline_alpha_deg", np.nan)) for entry in geofeatures],
        dtype=np.float64,
    )
    hough_peak_source = np.asarray(
        [str(entry.get("hough_peak_source", "houghpeaks")) for entry in geofeatures],
        dtype=str,
    )
    if args.hough_localmax_fallback:
        print(f"Conditional Hough localmax fallback patched {int(np.sum(hough_fallback_used))}/{len(hough_fallback_used)} frames")
    fascicle_persistence = select_fascicle_candidate_persistence(
        geofeatures,
        timtrack_alpha_raw,
        config=FascicleCandidatePersistenceConfig(
            enabled=bool(args.candidate_persistence),
            angle_min_deg=fas_angle_min,
            angle_max_deg=fas_angle_max,
            max_angle_step_deg=float(args.max_angle_step),
            hough_weight_bonus_deg=float(args.candidate_weight_bonus),
        ),
    )
    timtrack_alpha = np.asarray(fascicle_persistence["selected_alpha_deg"], dtype=np.float64)
    confidence_arrays: dict[str, np.ndarray] = {}
    compute_confidence = bool(args.adaptive_r or args.confidence_debug or args.save_confidence_plots or args.debug_detections)
    if compute_confidence:
        print("\nComputing ultrasound confidence metrics...")
        confidence_frames = read_gray_frames(args.video, n)
        confidence_arrays = compute_confidence_series(
            confidence_frames,
            rois,
            geofeatures,
            fascicle_prior,
            timtrack_alpha,
            config=SpeckleConfidenceConfig(),
            progress_every=args.progress_every if args.confidence_debug else None,
        )
    print(f"\nRunning 2-state fascicle Kalman ({args.kalman_mode})...")
    mm_per_pixel_arg = mm_per_px if np.isfinite(mm_per_px) else None
    super_lines_arr = np.asarray(apo_state["super_lines"], dtype=float)
    deep_lines_arr = np.asarray(apo_state["deep_lines"], dtype=float)
    final = run_fascicle_kalman_mode(
        args.kalman_mode,
        fascicle_prior,
        timtrack_alpha,
        super_lines_arr,
        deep_lines_arr,
        kalman_config,
        prediction_affine_matrices=np.asarray(klt["f_affine_matrices"], dtype=float),
        confidence_arrays=confidence_arrays,
        mm_per_pixel=mm_per_pixel_arg,
    )

    fixed_reference: dict[str, np.ndarray] | None = None
    comparison_rows: list[dict[str, float | int | str]] = []
    if args.compare_to_fixed_kalman:
        print("Running normal fixed-R Kalman for comparison...")
        fixed_reference = run_fascicle_kalman_mode(
            "fixed",
            fascicle_prior,
            timtrack_alpha,
            super_lines_arr,
            deep_lines_arr,
            kalman_config,
            prediction_affine_matrices=np.asarray(klt["f_affine_matrices"], dtype=float),
            mm_per_pixel=mm_per_pixel_arg,
        )
        comparison_rows = kalman_comparison_rows(final, fixed_reference)

    frames = np.arange(len(final["ANG_deg"]), dtype=np.int32)
    arrays: dict[str, np.ndarray] = {
        "frame": frames,
        "time_s": frames.astype(np.float64) / fps if fps and fps > 0 else np.full(len(frames), np.nan),
        "sup_apo_lines": np.asarray(apo_state["super_lines"], dtype=np.float64),
        "deep_apo_lines": np.asarray(apo_state["deep_lines"], dtype=np.float64),
        "klt_prior_segments": fascicle_prior,
        "klt_affine_ok": np.asarray(klt["f_affine_ok"], dtype=bool),
        "klt_affine_matrices": np.asarray(klt["f_affine_matrices"], dtype=np.float64),
        "klt_points_count": np.asarray(klt["f_points_count"], dtype=np.int32),
        "klt_tracked_count": np.asarray(klt["f_tracked_count"], dtype=np.int32),
        "klt_inlier_count": np.asarray(klt["f_inlier_count"], dtype=np.int32),
        "klt_tracker_redetected": np.asarray(klt["tracker_redetected"], dtype=bool),
        "klt_tracker_found_fraction": np.asarray(klt["tracker_found_fraction"], dtype=np.float64),
        "klt_tracker_state_points": np.asarray(klt["tracker_state_points"], dtype=object),
        "klt_tracked_old_points": np.asarray(klt["tracked_old_points"], dtype=object),
        "klt_tracked_new_points": np.asarray(klt["tracked_new_points"], dtype=object),
        "fascicle_segments": np.asarray(final["fascicle_segments"], dtype=np.float64),
        "fascicle_end_segments": np.asarray(final["fascicle_end_segments"], dtype=np.float64),
        "ANG_deg": np.asarray(final["ANG_deg"], dtype=np.float64),
        "PEN_deg": np.asarray(final["PEN_deg"], dtype=np.float64),
        "FL_px": np.asarray(final["FL_px"], dtype=np.float64),
        "timtrack_alpha_deg": timtrack_alpha,
        "raw_timtrack_alpha_deg": timtrack_alpha_raw,
        "hough_baseline_alpha_deg": hough_baseline_alpha,
        "hough_localmax_fallback_used": hough_fallback_used,
        "hough_localmax_fallback_mass_below_10deg": hough_fallback_mass,
        "hough_localmax_fallback_gap_to_lower_deg": hough_fallback_gap,
        "hough_peak_source": hough_peak_source,
        "fascicle_candidate_selected_idx": np.asarray(
            fascicle_persistence["selected_candidate_idx"],
            dtype=np.int32,
        ),
        "fascicle_candidate_raw_rejected": np.asarray(
            fascicle_persistence["raw_alpha_rejected"],
            dtype=bool,
        ),
        "fascicle_candidate_selection_reason": np.asarray(
            fascicle_persistence["selection_reason"],
            dtype=str,
        ),
        "selected_seed_segment": selected_seed,
        "selected_seed_alpha_deg": np.asarray(selected_alpha, dtype=np.float64),
        "mm_per_pixel": np.asarray(mm_per_px, dtype=np.float64),
        "measurement_mm_per_pixel": np.asarray(mm_per_px, dtype=np.float64),
        "apo_measurement_states": np.asarray(apo_state["measurement_states"], dtype=np.float64),
        "apo_accepted_measurement_states": np.asarray(apo_state["accepted_measurement_states"], dtype=np.float64),
        "apo_gating_r_scale": np.asarray(apo_state["gating_r_scale"], dtype=np.float64),
        "apo_rejected_endpoints": np.asarray(apo_state["rejected_endpoints"], dtype=bool),
        "apo_soft_downweighted_endpoints": np.asarray(apo_state["soft_downweighted_endpoints"], dtype=bool),
        "apo_line_rejected": np.asarray(apo_state["line_rejected"], dtype=bool),
        "apo_line_soft_downweighted": np.asarray(apo_state["line_soft_downweighted"], dtype=bool),
        "apo_gating_reasons": np.asarray(apo_state["gating_reasons"], dtype=str),
        "apo_consecutive_rejections": np.asarray(apo_state["consecutive_rejections"], dtype=np.int32),
    }
    add_kalman_state_arrays(arrays, final)
    if "FL_mm" in final:
        arrays["FL_mm"] = np.asarray(final["FL_mm"], dtype=np.float64)
    if fixed_reference is not None:
        arrays["fixed_fascicle_segments"] = np.asarray(fixed_reference["fascicle_segments"], dtype=np.float64)
        arrays["fixed_fascicle_end_segments"] = np.asarray(
            fixed_reference["fascicle_end_segments"],
            dtype=np.float64,
        )
        add_kalman_state_arrays(arrays, fixed_reference, prefix="fixed_")
        for key in ["ANG_deg", "PEN_deg", "FL_px", "FL_mm"]:
            if key not in final or key not in fixed_reference:
                continue
            fixed_key = f"fixed_{key}"
            delta_key = f"delta_vs_fixed_{key}"
            arrays[fixed_key] = np.asarray(fixed_reference[key], dtype=np.float64)
            arrays[delta_key] = np.asarray(final[key], dtype=np.float64) - arrays[fixed_key]
    if confidence_arrays:
        arrays.update(confidence_arrays)
    if "measurement_R_diag" in final:
        arrays["R_t_x_variance"] = np.asarray(final["measurement_R_diag"], dtype=np.float64)[:, 0]
        arrays["R_t_alpha_variance"] = np.asarray(final["measurement_R_diag"], dtype=np.float64)[:, 1]
        arrays["R_t_length_variance"] = arrays["R_t_x_variance"]
        arrays["R_t_theta_variance"] = arrays["R_t_alpha_variance"]
        arrays["kalman_measurement_r_scale"] = np.asarray(final["measurement_r_scale"], dtype=np.float64)
        arrays["kalman_measurement_r_scale_theta"] = np.asarray(
            final["measurement_r_scale_theta"], dtype=np.float64
        )
        arrays["kalman_measurement_r_scale_length"] = np.asarray(
            final["measurement_r_scale_length"], dtype=np.float64
        )

    npz_path = run_dir / f"{args.name}_strict_results.npz"
    np.savez_compressed(npz_path, **arrays)

    rows = []
    confidence_csv_columns = [
        "speckle_zncc",
        "speckle_confidence",
        "forward_backward_error",
        "valid_patch_fraction",
        "n_valid_patches",
        "n_total_patches",
        "motion_consistency",
        "motion_spread_px",
        "n_motion_points",
        "feature_reliability",
        "feature_peak_score",
        "feature_peak_count_score",
        "feature_mask_score",
        "feature_mask_density",
        "geometry_stability",
        "geometry_alpha_score",
        "geometry_pennation_score",
        "geometry_length_score",
        "geometry_angle_jump_deg",
        "geometry_angle_jump_score",
        "geometry_length_jump_px",
        "geometry_length_jump_score",
        "confidence_theta",
        "confidence_length",
        "combined_confidence",
        "r_scale",
        "r_scale_theta",
        "r_scale_length",
        "R_t_x_variance",
        "R_t_alpha_variance",
        "R_t_length_variance",
        "R_t_theta_variance",
        "kalman_measurement_r_scale",
        "kalman_measurement_r_scale_theta",
        "kalman_measurement_r_scale_length",
        "detection_success",
        "fixed_ANG_deg",
        "fixed_PEN_deg",
        "fixed_FL_px",
        "fixed_FL_mm",
        "delta_vs_fixed_ANG_deg",
        "delta_vs_fixed_PEN_deg",
        "delta_vs_fixed_FL_px",
        "delta_vs_fixed_FL_mm",
    ]
    for idx in range(len(frames)):
        row = {
            "Frame": int(frames[idx]),
            "Time": float(arrays["time_s"][idx]),
            "FL": float(arrays.get("FL_mm", arrays["FL_px"])[idx]),
            "PEN": float(arrays["PEN_deg"][idx]),
            "ANG": float(arrays["ANG_deg"][idx]),
            "FL_px": float(arrays["FL_px"][idx]),
            "TimTrackAlpha": float(timtrack_alpha[idx]),
            "RawTimTrackAlpha": float(timtrack_alpha_raw[idx]),
            "HoughBaselineAlpha": float(arrays["hough_baseline_alpha_deg"][idx]),
            "HoughPeakSource": str(arrays["hough_peak_source"][idx]),
            "HoughLocalmaxFallbackUsed": bool(arrays["hough_localmax_fallback_used"][idx]),
            "HoughFallbackMassBelow10Deg": float(arrays["hough_localmax_fallback_mass_below_10deg"][idx]),
            "HoughFallbackGapToLowerDeg": float(arrays["hough_localmax_fallback_gap_to_lower_deg"][idx]),
            "FascicleCandidateIdx": int(arrays["fascicle_candidate_selected_idx"][idx]),
            "FascicleRawRejected": bool(arrays["fascicle_candidate_raw_rejected"][idx]),
            "FascicleSelectionReason": str(arrays["fascicle_candidate_selection_reason"][idx]),
            "SuperApoRejected": bool(arrays["apo_line_rejected"][idx, 0]),
            "DeepApoRejected": bool(arrays["apo_line_rejected"][idx, 1]),
            "SuperApoSoftDownweighted": bool(arrays["apo_line_soft_downweighted"][idx, 0]),
            "DeepApoSoftDownweighted": bool(arrays["apo_line_soft_downweighted"][idx, 1]),
            "SuperApoGateReason": str(arrays["apo_gating_reasons"][idx, 0]),
            "DeepApoGateReason": str(arrays["apo_gating_reasons"][idx, 1]),
        }
        for column in confidence_csv_columns:
            if column not in arrays:
                continue
            value = arrays[column][idx]
            row[column] = bool(value) if np.asarray(value).dtype == np.dtype(bool) else float(value)
        rows.append(row)
    csv_path = run_dir / f"{args.name}_strict_FL_PEN_ANG.csv"
    write_csv(csv_path, rows)
    if args.print_time_series:
        print_time_series_rows(rows)

    comparison_csv_path: Optional[Path] = None
    if comparison_rows:
        comparison_csv_path = run_dir / f"{args.name}_kalman_comparison_vs_fixed.csv"
        write_csv(comparison_csv_path, comparison_rows)

    time_series_plot_path: Optional[Path] = None
    if not args.no_time_series_plot:
        time_series_plot_path = run_dir / f"{args.name}_ANG_PEN_FL_over_time.png"
        save_time_series_plot(time_series_plot_path, arrays)
    confidence_plot_path: Optional[Path] = None
    if args.save_confidence_plots and "combined_confidence" in arrays:
        confidence_plot_path = run_dir / f"{args.name}_confidence_diagnostics.png"
        save_confidence_plot(confidence_plot_path, arrays)

    debug_paths: dict[str, Path] = {}
    if args.debug_detections:
        debug_dir = run_dir / "debug"
        debug_specs = [
            (
                "fascicle_candidates",
                debug_dir / f"{args.name}_fascicle_candidate_lines.csv",
                list(fascicle_persistence["candidate_rows"]),
            ),
            (
                "fascicle_selection",
                debug_dir / f"{args.name}_fascicle_candidate_selection.csv",
                list(fascicle_persistence["selection_rows"]),
            ),
            (
                "aponeurosis_gating",
                debug_dir / f"{args.name}_aponeurosis_gating.csv",
                aponeurosis_gating_rows(apo_state, sup_meas, deep_meas),
            ),
            (
                "confidence_terms",
                debug_dir / f"{args.name}_confidence_terms.csv",
                confidence_debug_rows(confidence_arrays),
            ),
        ]
        for label, path, debug_rows in debug_specs:
            if not debug_rows:
                continue
            write_csv(path, debug_rows)
            debug_paths[label] = path

    metadata = {
        "video": str(args.video),
        "utt_export": str(args.utt_export),
        "roi_path": str(args.roi_path) if args.roi_path else None,
        "roi_parameter_update": not args.no_roi_parameter_update,
        "apo_fit_maxangle_super_deg": float(parms["apo"]["super"].get("maxangle", np.nan)),
        "apo_fit_maxangle_deep_deg": float(parms["apo"]["deep"].get("maxangle", np.nan)),
        "fas_angle_min_deg": fas_angle_min,
        "fas_angle_max_deg": fas_angle_max,
        "fas_angle_auto_enabled": bool(args.fas_angle_auto),
        "fas_angle_auto_applied": bool(fas_angle_auto_applied),
        "fas_angle_manual_override": bool(manual_fas_angle_override),
        "fas_angle_auto_candidates": (
            fascicle_angle_auto_diagnostics(fas_angle_auto_result, (fas_angle_min, fas_angle_max))
            if fas_angle_auto_result is not None
            else []
        ),
        "seed_angle_min_deg": float(seed_config.angle_min_deg),
        "seed_angle_max_deg": float(seed_config.angle_max_deg),
        "seed_angle_range_manual_override": args.seed_angle_range is not None,
        "seed_frames_requested": int(args.seed_frames),
        "seed_frames_used": int(seed_frame_count),
        "seed_selection_scale_source": "pixel_image_evidence",
        "seed_length_prior": "disabled_depth_independent",
        "hough_localmax_fallback": bool(args.hough_localmax_fallback),
        "hough_fallback_min_mass_below_10deg": float(args.hough_fallback_min_mass_below_10deg),
        "hough_fallback_min_gap_to_lower_deg": float(args.hough_fallback_min_gap_to_lower_deg),
        "hough_localmax_fallback_frames": int(np.sum(hough_fallback_used)),
        "candidate_persistence": bool(args.candidate_persistence),
        "max_angle_step_deg": float(args.max_angle_step),
        "candidate_weight_bonus_deg": float(args.candidate_weight_bonus),
        "apo_gating": bool(args.apo_gating),
        "apo_gate_mid_innovation_px": float(args.apo_gate_mid_innovation_px),
        "apo_gate_super_mid_jump_px": float(args.apo_gate_super_mid_jump_px),
        "apo_gate_deep_mid_jump_px": float(args.apo_gate_deep_mid_jump_px),
        "apo_gate_angle_jump_deg": float(args.apo_gate_angle_jump_deg),
        "apo_gate_max_rejections": int(args.apo_gate_max_rejections),
        "kalman_mode": args.kalman_mode,
        "adaptive_r": bool(args.adaptive_r),
        "compare_to_fixed_kalman": bool(args.compare_to_fixed_kalman),
        "annotate_kalman_comparison": bool(args.annotate_kalman_comparison),
        "kalman_comparison_csv": str(comparison_csv_path) if comparison_csv_path else None,
        "confidence_debug": bool(args.confidence_debug),
        "debug_detections": bool(args.debug_detections),
        "debug_csvs": {key: str(path) for key, path in debug_paths.items()},
        "frames": int(len(frames)),
        "fps": fps,
        "mm_per_pixel": mm_per_px,
        "measurement_mm_per_pixel": mm_per_px,
        "measurement_scale_source": measurement_scale_source,
        "matlab_export_image_depth_mm": matlab_image_depth_mm,
        "selected_seed_alpha_deg": selected_alpha,
        "selected_seed_segment": selected_seed.tolist(),
        "selected_cluster_id": selected_cluster,
        "runtime_s": time.time() - start_time,
    }
    metadata_path = run_dir / f"{args.name}_strict_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    overlay_paths = save_overlay_images(args.video, run_dir / "overlays", args.name, rois, arrays, args.save_overlays)
    annotated_path: Optional[Path] = None
    if not args.no_annotated_video:
        annotated_path = args.annotated_video or (run_dir / f"{args.name}_strict_annotated.mp4")
        save_annotated_video(
            args.video,
            annotated_path,
            rois,
            arrays,
            fps,
            show_kalman_comparison=bool(args.annotate_kalman_comparison),
        )

    print("\nDone.")
    print(f"CSV: {csv_path}")
    print(f"NPZ: {npz_path}")
    print(f"Metadata: {metadata_path}")
    if comparison_csv_path:
        print(f"Kalman comparison: {comparison_csv_path}")
    if time_series_plot_path:
        print(f"ANG/PEN/FL plot: {time_series_plot_path}")
    if confidence_plot_path:
        print(f"Confidence plot: {confidence_plot_path}")
    for label, path in debug_paths.items():
        print(f"Debug {label}: {path}")
    if annotated_path:
        print(f"Annotated video: {annotated_path}")
    for path in overlay_paths:
        print(f"Overlay image: {path}")

    return {
        "csv": csv_path,
        "npz": npz_path,
        "metadata": metadata_path,
        "kalman_comparison": comparison_csv_path,
        "time_series_plot": time_series_plot_path,
        "confidence_plot": confidence_plot_path,
        "annotated_video": annotated_path,
        "debug_dir": run_dir / "debug" if debug_paths else None,
    }


def main() -> None:
    args = parse_args()
    process_video(args)


if __name__ == "__main__":
    main()

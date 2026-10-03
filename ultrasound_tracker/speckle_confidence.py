"""Ultrasound-specific heuristic scores for adaptive Kalman measurement noise.

The functions in this module combine speckle coherence, local motion
consistency, feature-detector support, and anatomical geometry stability into
interpretable image/geometry scores.  Lower scores are mapped by design to
higher ``R_t`` and lower measurement gain.  Whether those scores are calibrated
to measurement error is an empirical question that must be tested against an
outcome which the adaptive weighting did not itself produce.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np


ROI = Tuple[int, int, int, int]


@dataclass(frozen=True)
class SpeckleConfidenceConfig:
    """Configuration for speckle/motion/feature confidence scoring."""

    block_size: int = 21
    stride: int = 24
    search_radius: int = 8
    min_texture_variance: float = 5.0
    zncc_low: float = 0.45
    zncc_high: float = 0.90
    confidence_floor: float = 0.05
    confidence_ceiling: float = 1.0
    r_min_scale: float = 0.5
    r_max_scale: float = 20.0
    r_gamma: float = 1.5
    use_zncc: bool = True
    min_points: int = 3
    motion_spread_scale_px: float = 3.0
    forward_backward_scale_px: float = 2.0
    min_feature_peaks_for_full_conf: int = 5
    feature_peak_scale: float = 25.0
    min_mask_density: float = 0.002
    max_mask_density: float = 0.35
    plausible_alpha_range_deg: Tuple[float, float] = (5.0, 85.0)
    plausible_pennation_range_deg: Tuple[float, float] = (0.0, 45.0)
    plausible_length_range_px: Tuple[float, float] = (20.0, 2000.0)
    angle_jump_scale_deg: float = 8.0
    length_jump_scale_px: float = 60.0
    weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "speckle": 0.35,
            "motion": 0.25,
            "feature": 0.25,
            "geometry": 0.15,
        }
    )
    theta_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "feature": 0.35,
            "motion": 0.25,
            "speckle": 0.20,
            "geometry_alpha": 0.10,
            "geometry_angle_jump": 0.10,
        }
    )
    length_weights: Mapping[str, float] = field(
        default_factory=lambda: {
            "geometry_length": 0.25,
            "aponeurosis": 0.25,
            "intersection": 0.20,
            "geometry_length_jump": 0.15,
            "geometry": 0.15,
        }
    )


@dataclass(frozen=True)
class ConfidenceMetrics:
    """Per-frame confidence summary used to adapt measurement covariance."""

    speckle_zncc: float = np.nan
    speckle_confidence: float = 1.0
    forward_backward_error: float = np.nan
    motion_consistency: float = 1.0
    feature_reliability: float = 1.0
    geometry_stability: float = 1.0
    confidence_theta: float = 1.0
    confidence_length: float = 1.0
    combined_confidence: float = 1.0
    r_scale: float = 1.0
    r_scale_theta: float = 1.0
    r_scale_length: float = 1.0
    detection_success: bool = True


def _gray_float(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image)
    if arr.ndim == 3:
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    if arr.ndim != 2:
        raise ValueError("image must be a 2D grayscale image or a BGR frame.")
    return arr.astype(np.float32, copy=False)


def _clip01(value: float, config: SpeckleConfidenceConfig) -> float:
    return float(np.clip(value, config.confidence_floor, config.confidence_ceiling))


def _roi_bounds(shape: Sequence[int], roi: Optional[ROI]) -> Tuple[int, int, int, int]:
    height, width = int(shape[0]), int(shape[1])
    if roi is None:
        return 0, 0, width, height
    x, y, w, h = [int(round(v)) for v in roi]
    x0 = int(np.clip(x, 0, width))
    y0 = int(np.clip(y, 0, height))
    x1 = int(np.clip(x + max(0, w), x0, width))
    y1 = int(np.clip(y + max(0, h), y0, height))
    return x0, y0, x1, y1


def _sample_grid_points(shape: Sequence[int], roi: Optional[ROI], config: SpeckleConfidenceConfig) -> np.ndarray:
    x0, y0, x1, y1 = _roi_bounds(shape, roi)
    margin = int(config.block_size // 2 + config.search_radius)
    xs = np.arange(x0 + margin, x1 - margin, max(1, int(config.stride)), dtype=np.float32)
    ys = np.arange(y0 + margin, y1 - margin, max(1, int(config.stride)), dtype=np.float32)
    if len(xs) == 0 or len(ys) == 0:
        return np.empty((0, 2), dtype=np.float32)
    xx, yy = np.meshgrid(xs, ys)
    return np.column_stack([xx.ravel(), yy.ravel()]).astype(np.float32)


def zncc(patch_a: np.ndarray, patch_b: np.ndarray, *, min_texture_variance: float = 0.0) -> float:
    """Zero-mean normalized cross-correlation in ``[-1, 1]``.

    Low-texture patches return ``NaN`` because correlation is not meaningful in
    near-homogeneous ultrasound regions.
    """

    a = np.asarray(patch_a, dtype=np.float32)
    b = np.asarray(patch_b, dtype=np.float32)
    if a.shape != b.shape:
        raise ValueError("patch_a and patch_b must have the same shape.")
    if a.size == 0:
        return float("nan")
    if float(np.var(a)) < float(min_texture_variance) or float(np.var(b)) < float(min_texture_variance):
        return float("nan")

    a0 = a - float(np.mean(a))
    b0 = b - float(np.mean(b))
    denom = float(np.sqrt(np.sum(a0 * a0) * np.sum(b0 * b0)))
    if denom <= 1e-12:
        return float("nan")
    return float(np.clip(np.sum(a0 * b0) / denom, -1.0, 1.0))


def _match_patch_at(
    source: np.ndarray,
    target: np.ndarray,
    point: Sequence[float],
    config: SpeckleConfidenceConfig,
) -> Optional[tuple[np.ndarray, float]]:
    half = int(config.block_size) // 2
    radius = int(config.search_radius)
    x, y = np.rint(np.asarray(point, dtype=np.float32)).astype(int)
    height, width = source.shape

    if (
        x - half < 0
        or x + half + 1 > width
        or y - half < 0
        or y + half + 1 > height
        or x - half - radius < 0
        or x + half + radius + 1 > target.shape[1]
        or y - half - radius < 0
        or y + half + radius + 1 > target.shape[0]
    ):
        return None

    template = source[y - half : y + half + 1, x - half : x + half + 1]
    if float(np.var(template)) < float(config.min_texture_variance):
        return None

    sx0 = x - half - radius
    sy0 = y - half - radius
    search = target[sy0 : y + half + radius + 1, sx0 : x + half + radius + 1]
    if search.shape[0] < config.block_size or search.shape[1] < config.block_size:
        return None

    if config.use_zncc:
        response = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        _, score, _, best_loc = cv2.minMaxLoc(response)
    else:
        response = cv2.matchTemplate(search, template, cv2.TM_CCORR_NORMED)
        _, score, _, best_loc = cv2.minMaxLoc(response)

    matched = np.asarray([sx0 + best_loc[0] + half, sy0 + best_loc[1] + half], dtype=np.float32)
    return matched, float(score)


def compute_speckle_coherence(
    prev_frame: np.ndarray,
    curr_frame: np.ndarray,
    points_or_blocks: Optional[np.ndarray] = None,
    roi: Optional[ROI] = None,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> dict[str, Any]:
    """Estimate speckle coherence by local patch matching.

    Points are zero-based ``(x, y)`` coordinates.  If no points are supplied, a
    grid is sampled inside ``roi``.  The returned ``speckle_confidence`` is a
    clipped, normalized version of median ZNCC, down-weighted when only a small
    fraction of patches are valid.
    """

    cfg = config or SpeckleConfidenceConfig()
    if cfg.block_size <= 0 or cfg.block_size % 2 == 0:
        raise ValueError("block_size must be a positive odd integer.")
    if cfg.search_radius < 0:
        raise ValueError("search_radius must be non-negative.")

    prev = _gray_float(prev_frame)
    curr = _gray_float(curr_frame)
    if prev.shape != curr.shape:
        raise ValueError("prev_frame and curr_frame must have the same shape.")

    if points_or_blocks is None:
        points = _sample_grid_points(prev.shape, roi, cfg)
    else:
        points = np.asarray(points_or_blocks, dtype=np.float32).reshape(-1, 2)

    matched_prev: list[np.ndarray] = []
    matched_curr: list[np.ndarray] = []
    scores: list[float] = []
    fb_errors: list[float] = []

    for point in points:
        forward = _match_patch_at(prev, curr, point, cfg)
        if forward is None:
            continue
        curr_point, score = forward
        if not np.isfinite(score):
            continue

        reverse = _match_patch_at(curr, prev, curr_point, cfg)
        if reverse is not None:
            back_point, _ = reverse
            fb_errors.append(float(np.linalg.norm(back_point - point)))

        matched_prev.append(np.asarray(point, dtype=np.float32))
        matched_curr.append(curr_point)
        scores.append(float(score))

    n_total = int(len(points))
    n_valid = int(len(scores))
    if n_total == 0 or n_valid == 0:
        return {
            "speckle_zncc": float("nan"),
            "speckle_confidence": float(cfg.confidence_floor),
            "forward_backward_error": float("nan"),
            "valid_patch_fraction": 0.0,
            "n_valid_patches": 0,
            "n_total_patches": n_total,
            "points_prev": np.empty((0, 2), dtype=np.float32),
            "points_curr": np.empty((0, 2), dtype=np.float32),
            "displacements": np.empty((0, 2), dtype=np.float32),
        }

    score_arr = np.asarray(scores, dtype=np.float32)
    median_zncc = float(np.nanmedian(score_arr))
    zncc_conf = (median_zncc - float(cfg.zncc_low)) / max(float(cfg.zncc_high - cfg.zncc_low), 1e-12)
    valid_fraction = float(n_valid / n_total)
    speckle_conf = _clip01(float(np.clip(zncc_conf, 0.0, 1.0) * np.sqrt(valid_fraction)), cfg)

    prev_points = np.vstack(matched_prev).astype(np.float32)
    curr_points = np.vstack(matched_curr).astype(np.float32)
    return {
        "speckle_zncc": median_zncc,
        "speckle_confidence": speckle_conf,
        "forward_backward_error": float(np.nanmedian(fb_errors)) if fb_errors else float("nan"),
        "valid_patch_fraction": valid_fraction,
        "n_valid_patches": n_valid,
        "n_total_patches": n_total,
        "points_prev": prev_points,
        "points_curr": curr_points,
        "displacements": curr_points - prev_points,
    }


def _robust_mad(values: np.ndarray) -> float:
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    med = float(np.median(arr))
    return float(1.4826 * np.median(np.abs(arr - med)))


def _points_in_roi(points: np.ndarray, roi: Optional[ROI]) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if roi is None or arr.size == 0:
        return np.ones(len(arr), dtype=bool)
    x, y, w, h = [float(v) for v in roi]
    return (arr[:, 0] >= x) & (arr[:, 0] <= x + w) & (arr[:, 1] >= y) & (arr[:, 1] <= y + h)


def compute_motion_consistency(
    prev_points: np.ndarray,
    curr_points: np.ndarray,
    roi: Optional[ROI] = None,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> dict[str, float]:
    """Score coherence of local displacement vectors in ``[0, 1]``."""

    cfg = config or SpeckleConfidenceConfig()
    prev = np.asarray(prev_points, dtype=np.float32).reshape(-1, 2)
    curr = np.asarray(curr_points, dtype=np.float32).reshape(-1, 2)
    n = min(len(prev), len(curr))
    prev = prev[:n]
    curr = curr[:n]
    keep = _points_in_roi(prev, roi) & _points_in_roi(curr, roi)
    prev = prev[keep]
    curr = curr[keep]

    if len(prev) < int(cfg.min_points):
        return {
            "motion_consistency": float(cfg.confidence_floor),
            "motion_spread_px": float("nan"),
            "median_dx_px": float("nan"),
            "median_dy_px": float("nan"),
            "n_motion_points": int(len(prev)),
        }

    displacements = curr - prev
    median_disp = np.median(displacements, axis=0)
    residuals = np.linalg.norm(displacements - median_disp, axis=1)
    spread = _robust_mad(residuals)
    confidence = float(np.exp(-spread / max(float(cfg.motion_spread_scale_px), 1e-12)))
    confidence = _clip01(confidence, cfg)
    return {
        "motion_consistency": confidence,
        "motion_spread_px": float(spread),
        "median_dx_px": float(median_disp[0]),
        "median_dy_px": float(median_disp[1]),
        "n_motion_points": int(len(displacements)),
    }


def _finite_array(value: Any) -> np.ndarray:
    arr = np.asarray(value, dtype=np.float64).reshape(-1)
    return arr[np.isfinite(arr)]


def _interval_score(value: float, lower: float, upper: float) -> float:
    if not np.isfinite(value):
        return 1.0
    if lower <= value <= upper:
        return 1.0
    span = max(float(upper - lower), 1e-6)
    distance = lower - value if value < lower else value - upper
    return float(np.exp(-distance / max(0.25 * span, 1e-6)))


def compute_feature_detection_reliability(
    detection_result: Mapping[str, Any] | None,
    mask: Optional[np.ndarray] = None,
    frame: Optional[np.ndarray] = None,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> dict[str, float | bool]:
    """Estimate Hough/Frangi/fascicle detector reliability from available signals."""

    del frame
    cfg = config or SpeckleConfidenceConfig()
    result = detection_result or {}
    alpha_values = _finite_array(result.get("alphas", []))
    weights = _finite_array(result.get("ws", result.get("weights", [])))
    alpha = _finite_array(result.get("alpha", []))
    detection_success = bool(result.get("detection_success", True))
    detection_success = detection_success and (alpha.size > 0 or alpha_values.size > 0)

    if not detection_success:
        return {
            "feature_reliability": float(cfg.confidence_floor),
            "detection_success": False,
            "feature_peak_score": 0.0,
            "feature_peak_count_score": 0.0,
            "feature_mask_score": 0.0,
        }

    n_peaks = int(alpha_values.size)
    peak_count_score = float(np.clip(n_peaks / max(int(cfg.min_feature_peaks_for_full_conf), 1), 0.0, 1.0))

    if weights.size:
        weights_sorted = np.sort(weights)[::-1]
        top = float(weights_sorted[0])
        peak_strength = float(np.clip(top / max(float(cfg.feature_peak_scale), 1e-12), 0.0, 1.0))
        if len(weights_sorted) > 1 and top > 0:
            separation = float(np.clip((top - float(weights_sorted[1])) / top, 0.0, 1.0))
        else:
            separation = 1.0
        peak_score = 0.75 * peak_strength + 0.25 * separation
    else:
        peak_score = 0.6 if n_peaks else 0.0

    mask_arr = mask
    if mask_arr is None:
        mask_arr = result.get("fascicle_masked", result.get("mask", None))
    if mask_arr is not None:
        density = float(np.mean(np.asarray(mask_arr).astype(bool)))
        low_score = float(np.clip(density / max(float(cfg.min_mask_density), 1e-12), 0.0, 1.0))
        high_score = float(np.clip(float(cfg.max_mask_density) / max(density, 1e-12), 0.0, 1.0))
        mask_score = min(low_score, high_score)
    else:
        density = float("nan")
        mask_score = 1.0

    feature = float(np.clip(0.35 * peak_count_score + 0.45 * peak_score + 0.20 * mask_score, 0.0, 1.0))
    feature = _clip01(feature, cfg)
    return {
        "feature_reliability": feature,
        "detection_success": True,
        "feature_peak_score": float(peak_score),
        "feature_peak_count_score": float(peak_count_score),
        "feature_mask_score": float(mask_score),
        "feature_mask_density": density,
    }


def _segment_length(segment: Optional[np.ndarray]) -> float:
    if segment is None:
        return float("nan")
    arr = np.asarray(segment, dtype=np.float64).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return float("nan")
    return float(np.hypot(arr[2] - arr[0], arr[3] - arr[1]))


def _segment_angle_deg(segment: Optional[np.ndarray]) -> float:
    if segment is None:
        return float("nan")
    arr = np.asarray(segment, dtype=np.float64).reshape(-1)
    if arr.size < 4 or not np.all(np.isfinite(arr[:4])):
        return float("nan")
    return float(abs(np.rad2deg(np.arctan2(arr[3] - arr[1], arr[2] - arr[0]))))


def _angle_delta_deg(current: float, previous: float) -> float:
    if not np.isfinite(current) or not np.isfinite(previous):
        return float("nan")
    return float(abs(((current - previous + 90.0) % 180.0) - 90.0))


def compute_geometry_stability(
    *,
    alpha_deg: Optional[float] = None,
    pennation_deg: Optional[float] = None,
    fascicle_length_px: Optional[float] = None,
    segment: Optional[np.ndarray] = None,
    previous_alpha_deg: Optional[float] = None,
    previous_length_px: Optional[float] = None,
    previous_segment: Optional[np.ndarray] = None,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> dict[str, float]:
    """Score anatomical plausibility and temporal stability in ``[0, 1]``."""

    cfg = config or SpeckleConfidenceConfig()
    alpha = float(alpha_deg) if alpha_deg is not None else _segment_angle_deg(segment)
    length = float(fascicle_length_px) if fascicle_length_px is not None else _segment_length(segment)
    prev_alpha = float(previous_alpha_deg) if previous_alpha_deg is not None else _segment_angle_deg(previous_segment)
    prev_length = float(previous_length_px) if previous_length_px is not None else _segment_length(previous_segment)

    alpha_score = _interval_score(alpha, *cfg.plausible_alpha_range_deg)
    pen_score = _interval_score(float(pennation_deg), *cfg.plausible_pennation_range_deg) if pennation_deg is not None else 1.0
    length_score = _interval_score(length, *cfg.plausible_length_range_px)

    angle_jump = _angle_delta_deg(alpha, prev_alpha)
    angle_jump_score = (
        float(np.exp(-angle_jump / max(float(cfg.angle_jump_scale_deg), 1e-12)))
        if np.isfinite(angle_jump)
        else 1.0
    )
    length_jump = abs(length - prev_length) if np.isfinite(length) and np.isfinite(prev_length) else float("nan")
    length_jump_score = (
        float(np.exp(-length_jump / max(float(cfg.length_jump_scale_px), 1e-12)))
        if np.isfinite(length_jump)
        else 1.0
    )

    scores = np.asarray([alpha_score, pen_score, length_score, angle_jump_score, length_jump_score], dtype=np.float64)
    scores = np.clip(scores[np.isfinite(scores)], cfg.confidence_floor, cfg.confidence_ceiling)
    stability = float(np.exp(np.mean(np.log(scores)))) if scores.size else 1.0
    stability = _clip01(stability, cfg)
    return {
        "geometry_stability": stability,
        "geometry_alpha_score": float(alpha_score),
        "geometry_pennation_score": float(pen_score),
        "geometry_length_score": float(length_score),
        "geometry_angle_jump_deg": float(angle_jump),
        "geometry_angle_jump_score": float(angle_jump_score),
        "geometry_length_jump_px": float(length_jump),
        "geometry_length_jump_score": float(length_jump_score),
    }


def _metric_value(metrics: Mapping[str, Any], key: str) -> float:
    aliases = {
        "speckle": "speckle_confidence",
        "motion": "motion_consistency",
        "feature": "feature_reliability",
        "geometry": "geometry_stability",
        "theta": "confidence_theta",
        "length": "confidence_length",
        "geometry_alpha": "geometry_alpha_score",
        "geometry_pennation": "geometry_pennation_score",
        "geometry_angle_jump": "geometry_angle_jump_score",
        "geometry_length": "geometry_length_score",
        "geometry_length_jump": "geometry_length_jump_score",
        "aponeurosis": "aponeurosis_stability",
        "intersection": "intersection_quality",
    }
    value = metrics.get(aliases.get(key, key), 1.0)
    try:
        value_f = float(value)
    except (TypeError, ValueError):
        value_f = 1.0
    return value_f if np.isfinite(value_f) else 1.0


def combine_confidence_metrics(
    metrics: ConfidenceMetrics | Mapping[str, Any],
    weights: Optional[Mapping[str, float]] = None,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> float:
    """Combine confidence components using a weighted geometric mean."""

    cfg = config or SpeckleConfidenceConfig()
    metric_map = metrics.__dict__ if isinstance(metrics, ConfidenceMetrics) else metrics
    active_weights = dict(weights or cfg.weights)
    total_weight = float(sum(max(float(w), 0.0) for w in active_weights.values()))
    if total_weight <= 0:
        return 1.0

    log_sum = 0.0
    for key, weight in active_weights.items():
        w = max(float(weight), 0.0)
        value = np.clip(_metric_value(metric_map, key), cfg.confidence_floor, cfg.confidence_ceiling)
        log_sum += w * float(np.log(value))
    return _clip01(float(np.exp(log_sum / total_weight)), cfg)


def combine_anisotropic_confidence_metrics(
    metrics: ConfidenceMetrics | Mapping[str, Any],
    theta_weights: Optional[Mapping[str, float]] = None,
    length_weights: Optional[Mapping[str, float]] = None,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> dict[str, float]:
    """Return separate angle and length confidence scores.

    The existing scalar confidence uses one weighted geometric mean,

    ``c = exp(sum_i w_i log(c_i) / sum_i w_i)``.

    This helper uses the same bounded geometric mean, but with two weight
    groups.  ``confidence_theta`` is intended for the orientation/pennation
    measurement variance and is driven mostly by Hough/Frangi feature support,
    local speckle coherence, and coherent fascicle motion.  ``confidence_length``
    is intended for the length-side measurement variance and is driven mostly
    by aponeurosis/intersection/geometry terms.  If optional keys such as
    ``aponeurosis_stability`` or ``intersection_quality`` are unavailable, they
    default to neutral confidence ``1.0`` so existing callers remain compatible.
    """

    cfg = config or SpeckleConfidenceConfig()
    metric_map = metrics.__dict__ if isinstance(metrics, ConfidenceMetrics) else metrics
    return {
        "confidence_theta": combine_confidence_metrics(
            metric_map,
            weights=theta_weights or cfg.theta_weights,
            config=cfg,
        ),
        "confidence_length": combine_confidence_metrics(
            metric_map,
            weights=length_weights or cfg.length_weights,
            config=cfg,
        ),
    }


def confidence_to_r_scale(confidence: float, config: Optional[SpeckleConfidenceConfig] = None) -> float:
    """Map confidence in ``[0, 1]`` to a bounded measurement-noise scale."""

    cfg = config or SpeckleConfidenceConfig()
    c = float(np.clip(confidence, cfg.confidence_floor, cfg.confidence_ceiling))
    scale = float(cfg.r_min_scale) + (float(cfg.r_max_scale) - float(cfg.r_min_scale)) * (1.0 - c) ** float(cfg.r_gamma)
    return float(np.clip(scale, min(cfg.r_min_scale, cfg.r_max_scale), max(cfg.r_min_scale, cfg.r_max_scale)))


def anisotropic_confidence_to_r_scales(
    confidence_theta: float,
    confidence_length: float,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> dict[str, float]:
    """Map theta and length confidence scores to bounded diagonal R scales.

    For each component, the scalar mapping is reused unchanged:

    ``s = r_min + (r_max - r_min) * (1 - c) ** r_gamma``.

    The returned scales are meant for a diagonal covariance
    ``diag([R_theta_0 * s_theta, R_L_0 * s_length])``.  The current
    MATLAB-compatible Kalman implementation stores its internal diagonal as
    ``[length-side x-state, theta alpha-state]``, so callers should pass
    ``r_scale_length`` to the x/length side and ``r_scale_theta`` to alpha.
    """

    return {
        "r_scale_theta": confidence_to_r_scale(confidence_theta, config),
        "r_scale_length": confidence_to_r_scale(confidence_length, config),
    }


def adapt_anisotropic_measurement_covariance(
    R_theta_base: float,
    R_length_base: float,
    confidence_theta: float,
    confidence_length: float,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> np.ndarray:
    """Return ``diag([R_theta_0*s_theta, R_L_0*s_L])``.

    This is a small convenience wrapper for documentation and tests.  The
    Kalman filter itself uses the same scales, but its state order is
    length-side ``x`` followed by orientation ``alpha`` for MATLAB parity.
    """

    scales = anisotropic_confidence_to_r_scales(confidence_theta, confidence_length, config)
    return np.diag(
        [
            float(R_theta_base) * scales["r_scale_theta"],
            float(R_length_base) * scales["r_scale_length"],
        ]
    ).astype(np.float64)


def adapt_measurement_covariance(
    R_base: np.ndarray | float,
    confidence: float,
    config: Optional[SpeckleConfidenceConfig] = None,
) -> np.ndarray:
    """Return ``R_t = R_base * r_scale(confidence)`` with bounded scaling."""

    base = np.asarray(R_base, dtype=np.float64)
    return base * confidence_to_r_scale(confidence, config)

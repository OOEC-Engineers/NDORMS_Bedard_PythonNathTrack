"""
MATLAB-style aponeurosis Hough detection used by UltraTimTrack TimTrack.

This module contains the aponeurosis branch that was prototyped in Notebook 38:

    raw ultrasound image
    -> adaptive threshold
    -> superficial/deep vertical cuts
    -> MATLAB-style Hough line detection
    -> aponeurosis y-vectors and fitted line segments

It is intentionally separate from :mod:`aponeurosis_detector`, whose historical
default is a cropped-ROI Frangi detector. The functions here work on the full
frame and keep MATLAB one-based coordinate vectors where that makes parity
checks easier.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_fill_holes, rotate as scipy_rotate, uniform_filter

from .timtrack_hough import hough_peaks, matlab_hough_accumulator


CutRange = Tuple[float, float]


@dataclass(frozen=True)
class MatlabHoughAponeurosisConfig:
    """Configuration for the full-frame MATLAB-style aponeurosis Hough path."""

    apox_1b: Optional[np.ndarray] = None
    super_cut: CutRange = (0.0, 0.5)
    deep_cut: CutRange = (0.5, 1.0)
    threshold_sensitivity: float = 0.5
    threshold_block_size: int | tuple[int, ...] | None = None
    threshold_method: str = "mean"
    threshold_c: float = 0.0
    hough_theta_step_deg: float = 1.0
    horizontal_replacement_angle_deg: float = 5.0
    fit_method: str = "enforce_maxangle"
    maxangle_constraint_mode: str = "symmetric"
    super_maxangle: float = 0.5
    deep_maxangle: float = 0.5
    super_order: int = 1
    deep_order: int = 1
    apomargin: int = 20
    napo: int = 10


def _get_nested(mapping: Mapping[str, Any], path: Sequence[str], default: Any = None) -> Any:
    obj: Any = mapping
    for key in path:
        if not isinstance(obj, Mapping) or key not in obj:
            return default
        obj = obj[key]
    return obj


def _as_cut(value: Any, default: CutRange) -> CutRange:
    arr = np.asarray(value if value is not None else default, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return default
    return float(arr[0]), float(arr[1])


def matlab_round_positive(x: np.ndarray | float) -> np.ndarray:
    """
    MATLAB ``round`` for non-negative values, where halves round away from zero.

    Pixel coordinates and cut positions in this module are non-negative, so this
    compact form is enough for the UltraTimTrack aponeurosis path.
    """

    return np.floor(np.asarray(x, dtype=np.float64) + 0.5).astype(int)


def normalize_to_uint8_for_threshold(image: np.ndarray) -> np.ndarray:
    """Normalize an image to uint8 before applying OpenCV adaptive thresholding.

    Kept for older calibration notebooks. The production MATLAB compatibility
    path below now ports MATLAB ``adaptthresh``/``imbinarize`` directly.
    """

    arr = np.asarray(image, dtype=np.float32)
    mn = float(np.nanmin(arr))
    mx = float(np.nanmax(arr))
    if not np.isfinite(mx - mn) or mx <= mn:
        return np.zeros(arr.shape, dtype=np.uint8)
    return np.clip((arr - mn) / (mx - mn) * 255.0, 0, 255).astype(np.uint8)


def _matlab_adaptthresh_neighborhood(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return MATLAB ``adaptthresh`` default neighborhood for an image shape."""

    return tuple((2 * np.floor(np.asarray(shape, dtype=np.float64) / 16.0) + 1).astype(int))


def _coerce_neighborhood_size(
    block_size: int | tuple[int, ...] | None,
    shape: tuple[int, ...],
) -> tuple[int, ...]:
    if block_size is None:
        return _matlab_adaptthresh_neighborhood(shape)
    if np.isscalar(block_size):
        values = np.repeat(int(block_size), len(shape))
    else:
        values = np.asarray(block_size, dtype=int).reshape(-1)
        if values.size == 1:
            values = np.repeat(int(values[0]), len(shape))
        if values.size != len(shape):
            raise ValueError("block_size must be scalar or match image dimensionality.")
    values = values.copy()
    values[values < 1] = 1
    values[values % 2 == 0] += 1
    return tuple(int(v) for v in values)


def _im2double_matlab_like(image: np.ndarray) -> np.ndarray:
    """Subset of MATLAB ``im2double`` behavior needed by ``adaptthresh``."""

    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return arr.astype(np.float64) / 255.0
    if arr.dtype == np.uint16:
        return arr.astype(np.float64) / 65535.0
    if arr.dtype == np.uint32:
        return arr.astype(np.float64) / 4294967295.0
    if arr.dtype == np.int8:
        return (arr.astype(np.float64) + 128.0) / 255.0
    if arr.dtype == np.int16:
        return (arr.astype(np.float64) + 32768.0) / 65535.0
    if arr.dtype == np.int32:
        return (arr.astype(np.float64) + 2147483648.0) / 4294967295.0
    return arr.astype(np.float64, copy=False)


def _class_range_matlab_like(image: np.ndarray) -> tuple[float, float]:
    arr = np.asarray(image)
    if arr.dtype == np.uint8:
        return 0.0, 255.0
    if arr.dtype == np.uint16:
        return 0.0, 65535.0
    if arr.dtype == np.uint32:
        return 0.0, 4294967295.0
    if arr.dtype == np.int8:
        return -128.0, 127.0
    if arr.dtype == np.int16:
        return -32768.0, 32767.0
    if arr.dtype == np.int32:
        return -2147483648.0, 2147483647.0
    return 0.0, 1.0


def _adaptthresh_scale_factor(sensitivity: float, foreground_polarity: str) -> float:
    sens = float(sensitivity)
    if not 0.0 <= sens <= 1.0:
        raise ValueError("sensitivity must be in [0, 1].")
    polarity = foreground_polarity.lower()
    if polarity == "bright":
        return 0.6 + (1.0 - sens)
    if polarity == "dark":
        return 0.4 + sens
    raise ValueError("foreground_polarity must be 'bright' or 'dark'.")


def adaptive_threshold_matlab_style(
    image: np.ndarray,
    *,
    sensitivity: float = 0.5,
    block_size: int | tuple[int, ...] | None = None,
    method: str = "mean",
    c: float = 0.0,
    foreground_polarity: str = "bright",
) -> np.ndarray:
    """
    Port MATLAB ``imbinarize(I,'adaptive','Sensitivity',s)`` for mean statistic.

    MATLAB first computes ``adaptthresh`` on ``im2double(I)`` with the default
    neighborhood ``2*floor(size(I)/16)+1`` when no neighborhood is supplied.
    Integer images are then compared against the threshold scaled back to the
    native class range, while floating point images are compared directly.
    """
    del c

    arr = np.asarray(image)
    if arr.ndim not in {2, 3}:
        raise ValueError("image must be a 2D image or 3D volume.")

    method_key = method.lower()
    if method_key != "mean":
        raise NotImplementedError("Only MATLAB adaptthresh Statistic='mean' is ported.")

    nhood = _coerce_neighborhood_size(block_size, arr.shape)
    scale_factor = _adaptthresh_scale_factor(float(sensitivity), foreground_polarity)
    image_double = _im2double_matlab_like(arr)
    local_mean = uniform_filter(image_double, size=nhood, mode="nearest")
    threshold = np.clip(scale_factor * local_mean, 0.0, 1.0)

    polarity = foreground_polarity.lower()
    if np.issubdtype(arr.dtype, np.integer):
        low, high = _class_range_matlab_like(arr)
        threshold_native = low + (high - low) * threshold
        return arr > threshold_native if polarity == "bright" else arr < threshold_native
    return arr > threshold if polarity == "bright" else arr < threshold


def zero_outside_vertical_cut(mask: np.ndarray, cut: CutRange) -> np.ndarray:
    """
    Apply MATLAB TimTrack vertical cut fractions to a binary mask.

    The kept rows are approximately ``round(cut[0] * n)`` through
    ``round(cut[1] * n) - 1`` in zero-based NumPy coordinates.
    """

    out = np.asarray(mask, dtype=bool).copy()
    n_rows = out.shape[0]
    k1 = int(matlab_round_positive(float(cut[0]) * n_rows))
    k2 = int(matlab_round_positive(float(cut[1]) * n_rows))
    if k1 > 0:
        out[: min(k1, n_rows), :] = False
    start = max(k2 - 1, 0)
    if start < n_rows:
        out[start:, :] = False
    return out


def make_matlab_apox(width: int, *, apomargin: int = 20, napo: int = 10) -> np.ndarray:
    """Create one-based aponeurosis sample x-positions when MATLAB parms lack ``apox``."""

    width = int(width)
    if width <= 0:
        raise ValueError("width must be positive.")
    if width <= 2 * apomargin:
        apomargin = max(1, width // 10)
    apox = np.round(np.linspace(apomargin + 1, width - apomargin, int(napo)))
    return np.clip(apox, 1, width).astype(np.float64)


def _sind(x: np.ndarray | float) -> np.ndarray:
    return np.sin(np.deg2rad(x))


def _cosd(x: np.ndarray | float) -> np.ndarray:
    return np.cos(np.deg2rad(x))


def _atan2d(y: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    return np.rad2deg(np.arctan2(y, x))


def get_aponeurosis_line_hough_matlab_like(
    apo_mask: np.ndarray,
    apox_1b: np.ndarray,
    line_type: str,
    *,
    theta_step_deg: float = 1.0,
    horizontal_replacement_angle_deg: float = 5.0,
) -> Tuple[np.ndarray, dict]:
    """
    Port of UltraTimTrack's nested ``get_apo_line`` Hough branch.

    Parameters
    ----------
    apo_mask:
        Binary mask after the superficial/deep cut.
    apox_1b:
        MATLAB-style one-based x sample positions.
    line_type:
        ``"super"`` or ``"deep"``.

    Returns
    -------
    apoy_1b, debug
        The detected aponeurosis y-vector in one-based row coordinates and a
        debug dictionary containing the Hough peak information.
    """

    bw = np.asarray(apo_mask, dtype=bool)
    if bw.ndim != 2:
        raise ValueError("apo_mask must be a 2D array.")

    apox_1b = np.asarray(apox_1b, dtype=np.float64).reshape(-1)
    theta_vec = np.arange(-90.0, 90.0, float(theta_step_deg), dtype=np.float64)
    hmat, theta, rho = matlab_hough_accumulator(bw, theta_vec, rho_resolution=1.0)

    rot_angle = float(horizontal_replacement_angle_deg)
    rotated = scipy_rotate(
        bw.astype(np.uint8),
        angle=rot_angle,
        reshape=False,
        order=0,
        mode="constant",
        cval=0,
        prefilter=False,
    ).astype(bool)
    hmat_rot, _, _ = matlab_hough_accumulator(rotated, [90.0 - rot_angle], rho_resolution=1.0)
    horizontal_cols = np.where(theta == -90.0)[0]
    if len(horizontal_cols):
        hmat[:, horizontal_cols[0]] = hmat_rot[:, 0]

    peaks = hough_peaks(hmat, 1, threshold=0.0, theta_degrees=theta)
    if len(peaks) == 0:
        return np.full_like(apox_1b, np.nan), {
            "theta": np.nan,
            "rho": np.nan,
            "peak": np.asarray([], dtype=np.int64),
            "hmat": hmat,
        }

    peak_row, peak_col = peaks[0]
    peak_rho = float(rho[peak_row])
    peak_theta = float(theta[peak_col])

    x0 = apox_1b - 1.0
    if abs(float(_sind(peak_theta))) < 1e-12:
        y1 = np.full_like(x0, np.nan)
    else:
        y0 = np.rint((peak_rho - x0 * _cosd(peak_theta)) / _sind(peak_theta))
        y1 = y0 + 1.0

    if peak_theta == -90.0:
        theta_rot = peak_theta + rot_angle
        yi = -np.rint((peak_rho - apox_1b * _cosd(theta_rot)) / _sind(theta_rot))
        n_rows, n_cols = bw.shape
        center = np.rint([n_cols / 2.0, n_rows / 2.0])
        shifted = np.vstack([apox_1b - center[0], yi - center[1]])
        rotation = np.array(
            [[_cosd(rot_angle), -_sind(rot_angle)], [_sind(rot_angle), _cosd(rot_angle)]],
            dtype=np.float64,
        )
        transformed = rotation @ shifted
        translated = np.vstack([transformed[0] + center[0], transformed[1] + center[1]])
        y1 = np.rint(translated[1])

    filled = binary_fill_holes(bw)
    n_rows, n_cols = filled.shape
    apoy = np.full_like(apox_1b, np.nan, dtype=np.float64)
    kind = line_type.lower()

    for i, (x_1b, y_1b) in enumerate(zip(apox_1b, y1)):
        if not np.isfinite(x_1b) or not np.isfinite(y_1b):
            continue
        col = int(x_1b) - 1
        row = int(y_1b) - 1
        if col < 0 or col >= n_cols or row < 0 or row >= n_rows:
            continue
        if kind == "super":
            row_next = min(row + 1, n_rows - 1)
            if filled[row, col] or filled[row_next, col]:
                search = np.where(~filled[row_next:, col])[0]
                if len(search):
                    apoy[i] = (row_next + search[0]) + 1
            else:
                apoy[i] = row + 1
        elif kind == "deep":
            row_prev = max(row - 1, 0)
            if filled[row, col] or filled[row_prev, col]:
                search = np.where(~filled[:row, col])[0]
                if len(search):
                    apoy[i] = search[-1] + 2
            else:
                apoy[i] = row + 1
        else:
            raise ValueError("line_type must be 'super' or 'deep'.")

    return apoy, {
        "theta": peak_theta,
        "rho": peak_rho,
        "peak": peaks[0],
        "hmat": hmat,
        "theta_values": theta,
        "rho_values": rho,
    }


def fit_apo_matlab_like(
    apox_1b: np.ndarray,
    apoy_1b: np.ndarray,
    *,
    fit_method: str = "enforce_maxangle",
    maxangle: float = 0.5,
    order: int = 1,
    constraint_mode: str = "symmetric",
) -> Optional[np.ndarray]:
    """Fit an aponeurosis vector with a selectable max-angle constraint.

    ``symmetric`` preserves NathTrack's validated historical interpretation.
    ``matlab-one-sided`` reproduces UltraTimTrack ``fit_apo.m`` exactly: the
    constraint is applied only when ``fit_angle > maxangle``.  A negative
    ``fit_angle`` is therefore not clipped, regardless of its magnitude.
    """

    x = np.asarray(apox_1b, dtype=np.float64).reshape(-1)
    y = np.asarray(apoy_1b, dtype=np.float64).reshape(-1)
    valid = np.isfinite(x) & np.isfinite(y)
    if int(order) < 0:
        raise ValueError("order must be non-negative.")
    if np.sum(valid) < int(order) + 1:
        return None

    coef = np.polyfit(x[valid], y[valid], int(order)).astype(np.float64)
    if int(order) == 1 and fit_method == "enforce_maxangle":
        slope, intercept = coef
        fit_angle = -float(_atan2d(slope, 1.0))
        angle_limit = abs(float(maxangle))
        mode = str(constraint_mode).lower()
        if mode not in {"symmetric", "matlab-one-sided"}:
            raise ValueError("constraint_mode must be 'symmetric' or 'matlab-one-sided'.")
        should_constrain = (
            abs(fit_angle) > angle_limit
            if mode == "symmetric"
            else fit_angle > float(maxangle)
        )
        if np.isfinite(angle_limit) and should_constrain:
            clipped_angle = (
                float(np.clip(fit_angle, -angle_limit, angle_limit))
                if mode == "symmetric"
                else float(maxangle)
            )
            slope = -float(np.tan(np.deg2rad(clipped_angle)))
            intercept = float(np.mean(y[valid] - slope * x[valid]))
            coef = np.array([slope, intercept], dtype=np.float64)
    return coef


def line_segment_from_polyfit_1b(coef: Optional[np.ndarray], width: int) -> Optional[np.ndarray]:
    """
    Convert a one-based MATLAB ``polyfit`` line into a zero-based image segment.

    The returned segment has shape ``(4,)`` as ``[x1, y1, x2, y2]`` and is ready
    for :mod:`ultrasound_tracker.geometry` and ``final_outputs_from_lines``.
    """

    if coef is None:
        return None
    coef = np.asarray(coef, dtype=np.float64).reshape(-1)
    if coef.size != 2 or not np.all(np.isfinite(coef)):
        return None

    x_1b = np.array([1.0, float(width)], dtype=np.float64)
    y_1b = coef[0] * x_1b + coef[1]
    return np.array([0.0, y_1b[0] - 1.0, float(width - 1), y_1b[1] - 1.0], dtype=np.float64)


def detect_matlab_hough_aponeuroses(
    image: np.ndarray,
    *,
    config: Optional[MatlabHoughAponeurosisConfig] = None,
    parms: Optional[Mapping[str, Any]] = None,
) -> dict:
    """
    Detect superficial and deep aponeuroses with the Notebook 38 MATLAB path.

    ``parms`` may be the MATLAB/UltraTimTrack ``parms`` mapping. Values found in
    it override the dataclass defaults for ``apox``, cuts, fit settings, and
    threshold sensitivity, while the Notebook 38 threshold rule remains mean/71/0
    unless explicitly changed through ``config``.
    """

    img = np.asarray(image)
    if img.ndim != 2:
        raise ValueError("image must be a 2D grayscale frame.")

    cfg = config or MatlabHoughAponeurosisConfig()
    n_rows, n_cols = img.shape

    if parms is not None:
        apo_parms = parms.get("apo", {}) if isinstance(parms, Mapping) else {}
        apox_value = apo_parms.get("apox", cfg.apox_1b)
        apox = (
            np.asarray(apox_value, dtype=np.float64).reshape(-1)
            if apox_value is not None
            else np.asarray([], dtype=np.float64)
        )
        if apox.size == 0 or not np.any(np.isfinite(apox)):
            apox = make_matlab_apox(n_cols, apomargin=cfg.apomargin, napo=cfg.napo)
        super_cut = _as_cut(_get_nested(apo_parms, ("super", "cut")), cfg.super_cut)
        deep_cut = _as_cut(_get_nested(apo_parms, ("deep", "cut")), cfg.deep_cut)
        sensitivity = float(apo_parms.get("th", cfg.threshold_sensitivity))
        super_fit = _get_nested(apo_parms, ("super", "fit_method"), cfg.fit_method)
        deep_fit = _get_nested(apo_parms, ("deep", "fit_method"), cfg.fit_method)
        super_maxangle = float(_get_nested(apo_parms, ("super", "maxangle"), cfg.super_maxangle))
        deep_maxangle = float(_get_nested(apo_parms, ("deep", "maxangle"), cfg.deep_maxangle))
        super_order = int(_get_nested(apo_parms, ("super", "order"), cfg.super_order))
        deep_order = int(_get_nested(apo_parms, ("deep", "order"), cfg.deep_order))
        maxangle_constraint_mode = str(
            apo_parms.get("maxangle_constraint_mode", cfg.maxangle_constraint_mode)
        )
    else:
        apox = (
            np.asarray(cfg.apox_1b, dtype=np.float64).reshape(-1)
            if cfg.apox_1b is not None
            else make_matlab_apox(n_cols, apomargin=cfg.apomargin, napo=cfg.napo)
        )
        super_cut = cfg.super_cut
        deep_cut = cfg.deep_cut
        sensitivity = cfg.threshold_sensitivity
        super_fit = cfg.fit_method
        deep_fit = cfg.fit_method
        super_maxangle = cfg.super_maxangle
        deep_maxangle = cfg.deep_maxangle
        super_order = cfg.super_order
        deep_order = cfg.deep_order
        maxangle_constraint_mode = cfg.maxangle_constraint_mode

    apo_thres = adaptive_threshold_matlab_style(
        img,
        sensitivity=sensitivity,
        block_size=cfg.threshold_block_size,
        method=cfg.threshold_method,
        c=cfg.threshold_c,
    )
    apo_super = zero_outside_vertical_cut(apo_thres, super_cut)
    apo_deep = zero_outside_vertical_cut(apo_thres, deep_cut)

    super_vec, super_debug = get_aponeurosis_line_hough_matlab_like(
        apo_super,
        apox,
        "super",
        theta_step_deg=cfg.hough_theta_step_deg,
        horizontal_replacement_angle_deg=cfg.horizontal_replacement_angle_deg,
    )
    deep_vec, deep_debug = get_aponeurosis_line_hough_matlab_like(
        apo_deep,
        apox,
        "deep",
        theta_step_deg=cfg.hough_theta_step_deg,
        horizontal_replacement_angle_deg=cfg.horizontal_replacement_angle_deg,
    )

    super_coef = fit_apo_matlab_like(
        apox,
        super_vec,
        fit_method=str(super_fit),
        maxangle=super_maxangle,
        order=super_order,
        constraint_mode=maxangle_constraint_mode,
    )
    deep_coef = fit_apo_matlab_like(
        apox,
        deep_vec,
        fit_method=str(deep_fit),
        maxangle=deep_maxangle,
        order=deep_order,
        constraint_mode=maxangle_constraint_mode,
    )
    super_coef_lin = fit_apo_matlab_like(
        apox,
        super_vec,
        fit_method=str(super_fit),
        maxangle=super_maxangle,
        order=1,
        constraint_mode=maxangle_constraint_mode,
    )
    deep_coef_lin = fit_apo_matlab_like(
        apox,
        deep_vec,
        fit_method=str(deep_fit),
        maxangle=deep_maxangle,
        order=1,
        constraint_mode=maxangle_constraint_mode,
    )

    super_line = line_segment_from_polyfit_1b(super_coef_lin, n_cols)
    deep_line = line_segment_from_polyfit_1b(deep_coef_lin, n_cols)
    super_angle = -float(_atan2d(super_coef_lin[0], 1.0)) if super_coef_lin is not None else np.nan
    deep_angle = -float(_atan2d(deep_coef_lin[0], 1.0)) if deep_coef_lin is not None else np.nan

    return {
        "method": "matlab_hough",
        "maxangle_constraint_mode": maxangle_constraint_mode,
        "image_shape": (n_rows, n_cols),
        "apox_1b": apox,
        "apo_thres": apo_thres,
        "apo_super": apo_super,
        "apo_deep": apo_deep,
        "super_vec_1b": super_vec,
        "deep_vec_1b": deep_vec,
        "super_vector_points_0b": np.column_stack([apox - 1.0, super_vec - 1.0]),
        "deep_vector_points_0b": np.column_stack([apox - 1.0, deep_vec - 1.0]),
        "super_coef_1b": super_coef,
        "deep_coef_1b": deep_coef,
        "super_coef_linear_1b": super_coef_lin,
        "deep_coef_linear_1b": deep_coef_lin,
        "super_line_0b": super_line,
        "deep_line_0b": deep_line,
        "super_apo_angle_deg": super_angle,
        "deep_apo_angle_deg": deep_angle,
        "super_debug": super_debug,
        "deep_debug": deep_debug,
        "cuts": {"super": super_cut, "deep": deep_cut},
        "threshold": {
            "sensitivity": sensitivity,
            "block_size": cfg.threshold_block_size,
            "method": cfg.threshold_method,
            "c": cfg.threshold_c,
        },
    }


class MatlabHoughAponeurosisDetector:
    """Reusable package wrapper for Notebook 38's aponeurosis Hough branch."""

    def __init__(
        self,
        *,
        config: Optional[MatlabHoughAponeurosisConfig] = None,
        parms: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.config = config or MatlabHoughAponeurosisConfig()
        self.parms = parms

    @classmethod
    def from_parms(
        cls,
        parms: Mapping[str, Any],
        *,
        config: Optional[MatlabHoughAponeurosisConfig] = None,
    ) -> "MatlabHoughAponeurosisDetector":
        """Construct a detector from exported UltraTimTrack ``parms``."""

        return cls(config=config, parms=parms)

    def detect_pair(self, image: np.ndarray) -> dict:
        """Detect both superficial and deep aponeuroses on a full grayscale frame."""

        return detect_matlab_hough_aponeuroses(image, config=self.config, parms=self.parms)

    def detect(self, image: np.ndarray, kind: str = "both") -> dict:
        """
        Detect aponeuroses, optionally returning a single aponeurosis view.

        ``kind='both'`` returns the full pair result. ``'superficial'``/``'super'``
        and ``'deep'`` return a smaller dictionary with the requested vector and
        line while preserving the full pair result under ``pair_result``.
        """

        pair = self.detect_pair(image)
        key = kind.lower()
        if key == "both":
            return pair
        if key in {"super", "superficial"}:
            return {
                "method": "matlab_hough",
                "kind": "superficial",
                "apox_1b": pair["apox_1b"],
                "vector_y_1b": pair["super_vec_1b"],
                "vector_points_0b": pair["super_vector_points_0b"],
                "fit_coef_1b": pair["super_coef_linear_1b"],
                "line_local": pair["super_line_0b"],
                "angle_deg": pair["super_apo_angle_deg"],
                "debug": pair["super_debug"],
                "pair_result": pair,
            }
        if key == "deep":
            return {
                "method": "matlab_hough",
                "kind": "deep",
                "apox_1b": pair["apox_1b"],
                "vector_y_1b": pair["deep_vec_1b"],
                "vector_points_0b": pair["deep_vector_points_0b"],
                "fit_coef_1b": pair["deep_coef_linear_1b"],
                "line_local": pair["deep_line_0b"],
                "angle_deg": pair["deep_apo_angle_deg"],
                "debug": pair["deep_debug"],
                "pair_result": pair,
            }
        raise ValueError("kind must be 'both', 'superficial', 'super', or 'deep'.")

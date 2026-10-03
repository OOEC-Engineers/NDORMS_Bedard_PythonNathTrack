"""Helpers for MATLAB TimTrack geofeature compatibility.

The raw ``dohough`` path returns an angle from one binary fascicle mask.  Saved
UltraTimTrack results also keep the peak angle stream under
``geofeatures(i).alphas`` and ``geofeatures(i).ws``.  MATLAB's saved
``geofeatures(i).alpha`` is the weighted median of those saved peaks.

Keeping this reconstruction explicit lets parity notebooks distinguish two
different gates:

* raw mask/doHough parity against intermediate MATLAB exports;
* saved full-sequence geofeature parity against ``Fdat`` / ``UTT`` outputs.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import convolve
from scipy.optimize import minimize_scalar

from .matlab_aponeurosis import (
    adaptive_threshold_matlab_style,
    detect_matlab_hough_aponeuroses,
    fit_apo_matlab_like,
    matlab_round_positive,
)
from .timtrack_hough import DoHoughParams, dohough, dohough_with_localmax_fallback, weighted_median


def _as_entry_list(geofeatures: object) -> list[Mapping]:
    if isinstance(geofeatures, Mapping):
        return [geofeatures]
    if isinstance(geofeatures, (list, tuple)):
        return [entry for entry in geofeatures if isinstance(entry, Mapping)]
    return [
        entry
        for entry in np.asarray(geofeatures, dtype=object).reshape(-1)
        if isinstance(entry, Mapping)
    ]


def alpha_from_saved_peaks(alphas: object, weights: object) -> float:
    """Return MATLAB's saved TimTrack alpha from saved peak angles and weights."""

    alpha_arr = np.asarray(alphas, dtype=np.float64).reshape(-1)
    weight_arr = np.asarray(weights, dtype=np.float64).reshape(-1)
    n = min(alpha_arr.size, weight_arr.size)
    if n == 0:
        return float("nan")

    alpha_arr = alpha_arr[:n]
    weight_arr = weight_arr[:n]
    valid = np.isfinite(alpha_arr) & np.isfinite(weight_arr) & (weight_arr > 0)
    if not np.any(valid):
        return float("nan")
    return weighted_median(alpha_arr[valid], weight_arr[valid])


def reconstruct_saved_geofeature_alpha(geofeatures: object) -> np.ndarray:
    """
    Reconstruct ``geofeatures.alpha`` from saved ``alphas`` and ``ws`` arrays.

    MATLAB stores the peak weights as ``ws`` in the full UTT numeric export.
    Some diagnostic exports use ``weights`` instead, so this helper accepts both
    names and prefers ``ws`` when available.
    """

    out = []
    for entry in _as_entry_list(geofeatures):
        weights = entry.get("ws", entry.get("weights", []))
        out.append(alpha_from_saved_peaks(entry.get("alphas", []), weights))
    return np.asarray(out, dtype=np.float64)


def extract_saved_peak_arrays(
    geofeatures: object,
    *,
    max_peaks: int = 10,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Return ``(alpha, peak_alphas, peak_weights)`` from saved geofeatures.

    Arrays are padded with NaNs to ``max_peaks`` columns.  ``alpha`` is
    reconstructed from the saved peaks, not copied from ``entry['alpha']``.
    """

    entries = _as_entry_list(geofeatures)
    peak_alphas = np.full((len(entries), int(max_peaks)), np.nan, dtype=np.float64)
    peak_weights = np.full_like(peak_alphas, np.nan)
    alpha = np.full(len(entries), np.nan, dtype=np.float64)

    for idx, entry in enumerate(entries):
        weights = np.asarray(entry.get("ws", entry.get("weights", [])), dtype=np.float64).reshape(-1)
        alphas = np.asarray(entry.get("alphas", []), dtype=np.float64).reshape(-1)
        n = min(int(max_peaks), len(alphas), len(weights))
        if n:
            peak_alphas[idx, :n] = alphas[:n]
            peak_weights[idx, :n] = weights[:n]
        alpha[idx] = alpha_from_saved_peaks(alphas, weights)

    return alpha, peak_alphas, peak_weights


def saved_alpha_error(saved_geofeatures: object, reference_alpha: Optional[Iterable[float]] = None) -> dict:
    """
    Summarize reconstruction error against an optional saved reference alpha.

    This is mainly a compact notebook helper.  When ``reference_alpha`` is not
    provided, the function compares against ``entry['alpha']`` if present.
    """

    entries = _as_entry_list(saved_geofeatures)
    estimate = reconstruct_saved_geofeature_alpha(entries)
    if reference_alpha is None:
        reference = np.asarray(
            [
                np.asarray(entry.get("alpha", np.nan), dtype=np.float64).reshape(-1)[0]
                for entry in entries
            ],
            dtype=np.float64,
        )
    else:
        reference = np.asarray(reference_alpha, dtype=np.float64).reshape(-1)

    n = min(len(reference), len(estimate))
    reference = reference[:n]
    estimate = estimate[:n]
    mask = np.isfinite(reference) & np.isfinite(estimate)
    diff = estimate[mask] - reference[mask]
    return {
        "n": int(np.sum(mask)),
        "max_abs_error_deg": float(np.nanmax(np.abs(diff))) if diff.size else np.nan,
        "rmse_deg": float(np.sqrt(np.nanmean(diff**2))) if diff.size else np.nan,
    }


def _sind(x: np.ndarray | float) -> np.ndarray:
    return np.sin(np.deg2rad(x))


def _cosd(x: np.ndarray | float) -> np.ndarray:
    return np.cos(np.deg2rad(x))


def _tand(x: np.ndarray | float) -> np.ndarray:
    return np.tan(np.deg2rad(x))


def _atan2d(y: np.ndarray | float, x: np.ndarray | float) -> np.ndarray:
    return np.rad2deg(np.arctan2(y, x))


def hessian2d_matlab_like(image: np.ndarray, sigma: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hessian filters matching the Notebook 38 MATLAB Frangi port."""

    sigma = float(sigma)
    radius = int(round(3 * sigma))
    coords = np.arange(-radius, radius + 1, dtype=np.float64)
    x_grid, y_grid = np.meshgrid(coords, coords, indexing="ij")
    common = np.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    dxx_kernel = (1 / (2 * np.pi * sigma**4)) * ((x_grid**2 / sigma**2) - 1) * common
    dxy_kernel = (1 / (2 * np.pi * sigma**6)) * (x_grid * y_grid) * common
    dyy_kernel = dxx_kernel.T
    img = np.asarray(image, dtype=np.float64)
    return (
        convolve(img, dxx_kernel, mode="nearest"),
        convolve(img, dxy_kernel, mode="nearest"),
        convolve(img, dyy_kernel, mode="nearest"),
    )


def eig2image_matlab_like(
    dxx: np.ndarray,
    dxy: np.ndarray,
    dyy: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Eigenvalue ordering used by the MATLAB Frangi implementation."""

    tmp = np.sqrt((dxx - dyy) ** 2 + 4 * dxy**2)
    v2x = 2 * dxy
    v2y = dyy - dxx + tmp
    mag = np.sqrt(v2x**2 + v2y**2)
    ok = mag != 0
    v2x = v2x.copy()
    v2y = v2y.copy()
    v2x[ok] /= mag[ok]
    v2y[ok] /= mag[ok]
    v1x = -v2y
    v1y = v2x
    mu1 = 0.5 * (dxx + dyy + tmp)
    mu2 = 0.5 * (dxx + dyy - tmp)
    check = np.abs(mu1) > np.abs(mu2)
    lambda1 = mu1.copy()
    lambda2 = mu2.copy()
    lambda1[check] = mu2[check]
    lambda2[check] = mu1[check]
    ix = v1x.copy()
    iy = v1y.copy()
    ix[check] = v2x[check]
    iy[check] = v2y[check]
    return lambda1, lambda2, ix, iy


def frangi_filter2d_matlab_like(image: np.ndarray, options: Mapping[str, Any]) -> np.ndarray:
    """Frangi filter branch used by MATLAB ``filter_usimage`` parity notebooks."""

    scale_range = np.asarray(options.get("FrangiScaleRange", [1, 10]), dtype=np.float64).reshape(-1)
    scale_ratio = float(options.get("FrangiScaleRatio", 2))
    beta_one = float(options.get("FrangiBetaOne", 0.5))
    beta_two = float(options.get("FrangiBetaTwo", 15))
    black_white = bool(options.get("BlackWhite", True))
    sigmas = np.sort(np.arange(scale_range[0], scale_range[1] + 1e-12, scale_ratio, dtype=np.float64))
    beta = 2 * beta_one**2
    c_value = 2 * beta_two**2
    img = np.asarray(image, dtype=np.float64)
    filtered_by_scale = []

    for sigma in sigmas:
        dxx, dxy, dyy = hessian2d_matlab_like(img, sigma)
        dxx *= sigma**2
        dxy *= sigma**2
        dyy *= sigma**2
        small_abs, large_abs, _, _ = eig2image_matlab_like(dxx, dxy, dyy)
        lambda2 = small_abs
        lambda1 = large_abs.copy()
        lambda1[lambda1 == 0] = np.finfo(float).eps
        rb = (lambda2 / lambda1) ** 2
        s2 = lambda1**2 + lambda2**2
        filtered = np.exp(-rb / beta) * (1 - np.exp(-s2 / c_value))
        if black_white:
            filtered[lambda1 < 0] = 0
        else:
            filtered[lambda1 > 0] = 0
        filtered_by_scale.append(filtered)

    if not filtered_by_scale:
        return np.zeros_like(img, dtype=np.float64)
    if len(filtered_by_scale) == 1:
        return filtered_by_scale[0]
    return np.max(np.stack(filtered_by_scale, axis=2), axis=2)


def matlab_literal_fascicle_subtraction(
    fascicle_threshold: np.ndarray,
    super_vec_1b: np.ndarray,
    deep_vec_1b: np.ndarray,
) -> np.ndarray:
    """
    Reproduce MATLAB's Hough branch ``fas_thres(super_obj | deep_obj) = 0``.

    In this branch ``super_obj`` and ``deep_obj`` are y-vectors, not masks, so
    MATLAB logical linear indexing removes a small prefix of the column-major
    flattened image.  It is kept because this is what matched the raw export.
    """

    out = np.asarray(fascicle_threshold, dtype=bool).copy()
    literal = np.asarray(super_vec_1b, dtype=bool) | np.asarray(deep_vec_1b, dtype=bool)
    flat = out.ravel(order="F")
    n = min(len(flat), len(literal))
    flat[:n][literal[:n]] = False
    return flat.reshape(out.shape, order="F")


def line_mask_fascicle_subtraction(
    fascicle_threshold: np.ndarray,
    apox_1b: np.ndarray,
    super_vec_1b: np.ndarray,
    deep_vec_1b: np.ndarray,
    *,
    radius: int = 2,
) -> np.ndarray:
    """Alternative spatial aponeurosis subtraction used for diagnostics."""

    out = np.asarray(fascicle_threshold, dtype=bool).copy()
    n_rows, n_cols = out.shape
    apox = np.asarray(apox_1b, dtype=np.float64).reshape(-1)
    for vec in [super_vec_1b, deep_vec_1b]:
        y = np.asarray(vec, dtype=np.float64).reshape(-1)
        valid = np.isfinite(apox) & np.isfinite(y)
        if np.sum(valid) < 2:
            continue
        coef = np.polyfit(apox[valid], y[valid], 1)
        xs = np.arange(1, n_cols + 1, dtype=np.float64)
        ys = coef[0] * xs + coef[1]
        for x_1b, y_1b in zip(xs, ys):
            col = int(round(x_1b)) - 1
            row = int(round(y_1b)) - 1
            if not 0 <= col < n_cols:
                continue
            r0 = max(0, row - int(radius))
            r1 = min(n_rows, row + int(radius) + 1)
            out[r0:r1, col] = False
    return out


def get_fascicle_mask_matlab_like(
    fascicle_threshold: np.ndarray,
    super_vec_1b: np.ndarray,
    deep_vec_1b: np.ndarray,
) -> tuple[np.ndarray, tuple[float, float], tuple[float, float]]:
    """Port of MATLAB ``get_fasMask`` ellipse construction."""

    binary = np.asarray(fascicle_threshold, dtype=bool)
    n_rows, n_cols = binary.shape
    mean_super = (
        matlab_round_positive(np.nanmean(super_vec_1b))
        if np.any(np.isfinite(super_vec_1b))
        else np.asarray(np.nan)
    )
    mean_deep = (
        matlab_round_positive(np.nanmean(deep_vec_1b))
        if np.any(np.isfinite(deep_vec_1b))
        else np.asarray(np.nan)
    )
    if not np.isfinite(mean_super) or not np.isfinite(mean_deep) or float(mean_deep) <= float(mean_super):
        return np.ones_like(binary), (n_rows / 2.0, n_cols / 2.0), (float(mean_super), float(mean_deep))

    r_vertical = (float(mean_deep) - float(mean_super)) / 2.0
    r_horizontal = n_cols / 2.0
    x_center = r_horizontal
    y_center = float(mean_super) + r_vertical
    x_grid, y_grid = np.meshgrid(np.arange(1, n_cols + 1), np.arange(1, n_rows + 1))
    mask = ((x_grid - x_center) / r_horizontal) ** 2 + ((y_grid - y_center) / r_vertical) ** 2 <= 1.0
    return mask, (r_vertical, r_horizontal), (float(mean_super), float(mean_deep))


def _parms_fascicle_mask(
    fas_parms: Mapping[str, Any],
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray] | None:
    """Return MATLAB's saved ``parms.fas.Emask`` when auto_ultrasound would reuse it."""

    if bool(fas_parms.get("redo_ROI", False)):
        return None
    if "Emask" not in fas_parms or "Emask_radius" not in fas_parms:
        return None

    emask = np.asarray(fas_parms["Emask"], dtype=bool)
    if emask.shape != shape:
        if emask.size != int(shape[0]) * int(shape[1]):
            return None
        emask = emask.reshape(shape, order="F")

    radius = np.asarray(fas_parms["Emask_radius"], dtype=np.float64).reshape(-1)
    if radius.size < 2 or not np.all(np.isfinite(radius[:2])):
        return None
    return emask, radius[:2]


def _polyval(coef: Optional[np.ndarray], x: object) -> np.ndarray:
    if coef is None:
        return np.full_like(np.asarray(x, dtype=np.float64), np.nan, dtype=np.float64)
    return np.polyval(np.asarray(coef, dtype=np.float64), x)


def _trim_trailing_nan_coefficients(coef: np.ndarray) -> np.ndarray:
    arr = np.asarray(coef, dtype=np.float64).reshape(-1)
    while arr.size and not np.isfinite(arr[-1]):
        arr = arr[:-1]
    return arr


def extrapolated_apo_x_matlab_like(
    super_coef_linear_1b: np.ndarray,
    deep_coef_linear_1b: np.ndarray,
    alpha_deg: float,
    width: int,
) -> tuple[float, np.ndarray]:
    """Port the MATLAB extrapolated fascicle/aponeurosis intersection search."""

    midpoint_x = round(width / 2)
    midpoint_y = np.mean([
        _polyval(deep_coef_linear_1b, midpoint_x),
        _polyval(super_coef_linear_1b, midpoint_x),
    ])
    fascicle_slope = -float(_tand(alpha_deg))

    def cost(intercept: float) -> float:
        deep_denom = deep_coef_linear_1b[0] - fascicle_slope
        super_denom = super_coef_linear_1b[0] - fascicle_slope
        if abs(deep_denom) < 1e-12 or abs(super_denom) < 1e-12:
            return np.inf
        x_deep = (intercept - deep_coef_linear_1b[1]) / deep_denom
        x_super = (intercept - super_coef_linear_1b[1]) / super_denom
        return max((midpoint_x - x_deep) ** 2, (midpoint_x - x_super) ** 2)

    span = max(width, 100)
    result = minimize_scalar(
        cost,
        bounds=(float(midpoint_y) - span, float(midpoint_y) + span),
        method="bounded",
        options={"xatol": 1e-6},
    )
    intercept = float(result.x) if result.success else float(midpoint_y - midpoint_x * fascicle_slope)
    denom = deep_coef_linear_1b[0] - fascicle_slope
    apo_x = (intercept - deep_coef_linear_1b[1]) / denom if abs(denom) > 1e-12 else np.nan
    return float(apo_x), np.asarray([fascicle_slope, intercept], dtype=np.float64)


def _polyline_intersection_x(poly_coef: np.ndarray, line_coef: np.ndarray, x_hint: float) -> float:
    poly = _trim_trailing_nan_coefficients(poly_coef)
    line = _trim_trailing_nan_coefficients(line_coef)
    if poly.size < 2 or line.size < 2:
        return np.nan

    if poly.size == 2:
        denom = line[0] - poly[0]
        if abs(denom) <= 1e-12:
            return np.nan
        return float((poly[1] - line[1]) / denom)

    line_padded = np.zeros_like(poly)
    line_padded[-2:] = line[:2]
    roots = np.roots(poly - line_padded)
    real_roots = np.asarray([root.real for root in roots if abs(root.imag) <= 1e-7], dtype=np.float64)
    real_roots = real_roots[np.isfinite(real_roots)]
    if real_roots.size == 0:
        return np.nan
    if np.isfinite(x_hint):
        return float(real_roots[int(np.argmin(np.abs(real_roots - x_hint)))])
    return float(real_roots[0])


def fascicle_segment_from_aponeuroses_and_alpha(
    super_coef_1b: np.ndarray,
    deep_coef_1b: np.ndarray,
    alpha_deg: float,
    width: int,
    *,
    super_coef_linear_1b: Optional[np.ndarray] = None,
    deep_coef_linear_1b: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Rebuild a MATLAB-style one-based fascicle segment for an explicit alpha.

    MATLAB's TimTrack path places the fascicle line by optimizing the line
    intercept between the superficial and deep aponeuroses, then intersects
    that line with the fitted aponeuroses.  This helper exposes the same
    geometry for diagnostics where the alpha stream is intentionally varied.
    """

    if not np.isfinite(alpha_deg):
        return np.full(4, np.nan, dtype=np.float64)

    super_coef = _trim_trailing_nan_coefficients(super_coef_1b)
    deep_coef = _trim_trailing_nan_coefficients(deep_coef_1b)
    super_linear = (
        _trim_trailing_nan_coefficients(super_coef_linear_1b)
        if super_coef_linear_1b is not None
        else super_coef
    )
    deep_linear = (
        _trim_trailing_nan_coefficients(deep_coef_linear_1b)
        if deep_coef_linear_1b is not None
        else deep_coef
    )
    if super_coef.size < 2 or deep_coef.size < 2 or super_linear.size < 2 or deep_linear.size < 2:
        return np.full(4, np.nan, dtype=np.float64)
    if not (
        np.all(np.isfinite(super_coef))
        and np.all(np.isfinite(deep_coef))
        and np.all(np.isfinite(super_linear[:2]))
        and np.all(np.isfinite(deep_linear[:2]))
    ):
        return np.full(4, np.nan, dtype=np.float64)

    apo_x, fas_coef = extrapolated_apo_x_matlab_like(super_linear[:2], deep_linear[:2], alpha_deg, width)
    fascicle_slope, fascicle_intercept = fas_coef

    super_denom = super_linear[0] - fascicle_slope
    super_hint = (
        (fascicle_intercept - super_linear[1]) / super_denom
        if abs(super_denom) > 1e-12
        else np.nan
    )
    x_super = _polyline_intersection_x(super_coef, fas_coef, super_hint)
    x_deep = _polyline_intersection_x(deep_coef, fas_coef, apo_x)
    if not np.isfinite(x_super) or not np.isfinite(x_deep):
        return np.full(4, np.nan, dtype=np.float64)

    y_super = np.polyval(fas_coef, x_super)
    y_deep = np.polyval(fas_coef, x_deep)
    return np.asarray([x_super, y_super, x_deep, y_deep], dtype=np.float64)


def filter_usimage_hough_matlab_like(
    image: np.ndarray,
    parms: Mapping[str, Any],
    *,
    subtraction_mode: str = "matlab_literal",
) -> dict:
    """Run the image threshold/Frangi portion of MATLAB ``filter_usimage``."""

    apo = detect_matlab_hough_aponeuroses(image, parms=parms)
    fas_parms = parms["fas"]
    fascicle_filtered = frangi_filter2d_matlab_like(np.asarray(image, dtype=np.float64), fas_parms["frangi"])
    fascicle_threshold = adaptive_threshold_matlab_style(
        fascicle_filtered,
        sensitivity=float(fas_parms.get("th", 0.5)),
        block_size=71,
        method="mean",
        c=0.0,
    )

    if subtraction_mode == "matlab_literal":
        fascicle_subtracted = matlab_literal_fascicle_subtraction(
            fascicle_threshold,
            apo["super_vec_1b"],
            apo["deep_vec_1b"],
        )
    elif subtraction_mode == "line_mask":
        fascicle_subtracted = line_mask_fascicle_subtraction(
            fascicle_threshold,
            apo["apox_1b"],
            apo["super_vec_1b"],
            apo["deep_vec_1b"],
        )
    else:
        raise ValueError("subtraction_mode must be 'matlab_literal' or 'line_mask'.")

    return {
        **apo,
        "fas_filt": fascicle_filtered,
        "fas_thres_raw": fascicle_threshold,
        "fas_thres": fascicle_subtracted,
    }


def detect_timtrack_geofeature_from_image(
    image: np.ndarray,
    parms: Mapping[str, Any],
    *,
    subtraction_mode: str = "matlab_literal",
    emask_mode: str = "matlab",
    hough_localmax_fallback: bool = False,
    hough_fallback_min_mass_below_10deg: float = 0.25,
    hough_fallback_min_gap_to_lower_deg: float = 4.0,
) -> dict:
    """
    Generate one TimTrack-like geofeature entry directly from an image frame.

    This is the independent Python image path.  It intentionally does not read
    MATLAB saved ``geofeatures``; those are used only as validation targets in
    notebooks.
    """

    image_arr = np.asarray(image)
    if image_arr.ndim != 2:
        raise ValueError("image must be a 2D grayscale frame.")
    n_rows, n_cols = image_arr.shape
    fas_parms = parms["fas"]
    apo_parms = parms["apo"]

    filtered = filter_usimage_hough_matlab_like(
        image_arr,
        parms,
        subtraction_mode=subtraction_mode,
    )
    dynamic_emask, dynamic_emask_radius, mean_depths = get_fascicle_mask_matlab_like(
        filtered["fas_thres"],
        filtered["super_vec_1b"],
        filtered["deep_vec_1b"],
    )
    emask_source = "dynamic"
    emask = dynamic_emask
    emask_radius = np.asarray(dynamic_emask_radius, dtype=np.float64)

    mode = str(emask_mode).lower()
    if mode not in {"matlab", "saved", "parms", "dynamic"}:
        raise ValueError("emask_mode must be 'matlab', 'saved', 'parms', or 'dynamic'.")
    if mode in {"matlab", "saved", "parms"}:
        saved_mask = _parms_fascicle_mask(fas_parms, (n_rows, n_cols))
        if saved_mask is not None:
            emask, emask_radius = saved_mask
            emask_source = "parms.fas.Emask"

    fascicle_masked = filtered["fas_thres"] & emask
    hough_params = DoHoughParams(
        houghangles=str(fas_parms["houghangles"]),
        angle_range=tuple(np.asarray(fas_parms["range"], dtype=np.float64).reshape(-1)),
        thetares=float(fas_parms["thetares"]),
        rhores=float(fas_parms["rhores"]),
        emask_radius=emask_radius,
        npeaks=int(fas_parms["npeaks"]),
        replace_diagonal_bias=True,
    )
    if hough_localmax_fallback:
        hough = dohough_with_localmax_fallback(
            fascicle_masked,
            hough_params,
            min_mass_below_10deg=float(hough_fallback_min_mass_below_10deg),
            min_gap_to_lower_deg=float(hough_fallback_min_gap_to_lower_deg),
        )
    else:
        hough = dohough(fascicle_masked, hough_params)
    alpha = float(hough["alpha"])

    apox = np.asarray(filtered["apox_1b"], dtype=np.float64).reshape(-1)
    super_settings = apo_parms.get("super", {}) if isinstance(apo_parms.get("super", {}), Mapping) else {}
    deep_settings = apo_parms.get("deep", {}) if isinstance(apo_parms.get("deep", {}), Mapping) else {}
    maxangle_constraint_mode = str(apo_parms.get("maxangle_constraint_mode", "symmetric"))
    super_coef = fit_apo_matlab_like(
        apox,
        filtered["super_vec_1b"],
        fit_method=str(super_settings.get("fit_method", "enforce_maxangle")),
        maxangle=float(super_settings.get("maxangle", 0.5)),
        order=int(super_settings.get("order", 1)),
        constraint_mode=maxangle_constraint_mode,
    )
    deep_coef = fit_apo_matlab_like(
        apox,
        filtered["deep_vec_1b"],
        fit_method=str(deep_settings.get("fit_method", "enforce_maxangle")),
        maxangle=float(deep_settings.get("maxangle", 0.5)),
        order=int(deep_settings.get("order", 1)),
        constraint_mode=maxangle_constraint_mode,
    )
    super_coef_lin = fit_apo_matlab_like(
        apox,
        filtered["super_vec_1b"],
        fit_method=str(super_settings.get("fit_method", "enforce_maxangle")),
        maxangle=float(super_settings.get("maxangle", 0.5)),
        order=1,
        constraint_mode=maxangle_constraint_mode,
    )
    deep_coef_lin = fit_apo_matlab_like(
        apox,
        filtered["deep_vec_1b"],
        fit_method=str(deep_settings.get("fit_method", "enforce_maxangle")),
        maxangle=float(deep_settings.get("maxangle", 0.5)),
        order=1,
        constraint_mode=maxangle_constraint_mode,
    )

    betha = -float(_atan2d(super_coef_lin[0], 1.0)) if super_coef_lin is not None else np.nan
    gamma = -float(_atan2d(deep_coef_lin[0], 1.0)) if deep_coef_lin is not None else np.nan

    if (
        bool(parms.get("extrapolation", 1))
        and np.isfinite(alpha)
        and super_coef_lin is not None
        and deep_coef_lin is not None
    ):
        apo_x, fas_coef = extrapolated_apo_x_matlab_like(super_coef_lin, deep_coef_lin, alpha, n_cols)
    else:
        apo_x = float(apo_parms.get("x", 20.0))
        fas_coef = np.asarray([np.nan, np.nan], dtype=np.float64)

    if (
        np.isfinite(apo_x)
        and super_coef is not None
        and deep_coef is not None
        and np.isfinite(betha)
    ):
        thickness = float((_polyval(deep_coef, apo_x) - _polyval(super_coef, apo_x)) * _cosd(betha))
    else:
        thickness = np.nan
    phi = float(alpha - betha) if np.isfinite(alpha) and np.isfinite(betha) else np.nan
    faslen = (
        float(thickness / _sind(phi))
        if np.isfinite(thickness) and np.isfinite(phi) and abs(float(_sind(phi))) > 1e-12
        else np.nan
    )
    extrapolated_fraction = (
        float((faslen - n_cols / _cosd(alpha)) / faslen)
        if np.isfinite(faslen) and faslen != 0 and np.isfinite(alpha)
        else np.nan
    )

    super_pos = _polyval(super_coef, [1.0, float(n_cols)])
    deep_pos = _polyval(deep_coef, [1.0, float(n_cols)])

    return {
        "alpha": alpha,
        "alphas": np.asarray(hough["alphas"], dtype=np.float64),
        "ws": np.asarray(hough["weights"], dtype=np.float64),
        "weights": np.asarray(hough["weights"], dtype=np.float64),
        "hs": np.asarray(hough["h_by_angle"], dtype=np.float64),
        "x": np.asarray(hough["X"], dtype=np.float64),
        "y": np.asarray(hough["Y"], dtype=np.float64),
        "phi": phi,
        "faslen": faslen,
        "betha": betha,
        "gamma": gamma,
        "thickness": thickness,
        "brightness": float(np.mean(image_arr)),
        "extrapolated_fraction": extrapolated_fraction,
        "super_pos": np.asarray(super_pos, dtype=np.float64),
        "deep_pos": np.asarray(deep_pos, dtype=np.float64),
        "super_coef": super_coef,
        "deep_coef": deep_coef,
        "super_coef_linear": super_coef_lin,
        "deep_coef_linear": deep_coef_lin,
        "super_vec_1b": np.asarray(filtered["super_vec_1b"], dtype=np.float64),
        "deep_vec_1b": np.asarray(filtered["deep_vec_1b"], dtype=np.float64),
        "fas_coef": fas_coef,
        "apo_x": apo_x,
        "Emask": emask,
        "Emask_radius": np.asarray(emask_radius, dtype=np.float64),
        "Emask_source": emask_source,
        "mean_depths": np.asarray(mean_depths, dtype=np.float64),
        "fascicle_masked": fascicle_masked,
        "filtered": filtered,
        "hough_result": hough,
        "hough_peak_source": str(hough.get("selected_peak_source", hough.get("peak_source", "houghpeaks"))),
        "hough_localmax_fallback_used": bool(hough.get("localmax_fallback_used", False)),
        "hough_localmax_fallback_mass_below_10deg": float(
            hough.get("localmax_fallback_mass_below_10deg", np.nan)
        ),
        "hough_localmax_fallback_gap_to_lower_deg": float(
            hough.get("localmax_fallback_gap_to_lower_deg", np.nan)
        ),
        "hough_baseline_alpha_deg": float(hough.get("baseline_alpha", alpha)),
        "image_shape": (n_rows, n_cols),
    }


def fascicle_segment_from_geofeature(
    entry: Mapping[str, Any],
    *,
    alpha_override: Optional[float] = None,
) -> np.ndarray:
    """
    Reconstruct a one-based fascicle segment from a TimTrack geofeature entry.

    The preferred reconstruction intersects ``fas_coef`` with the fitted
    superficial and deep aponeurosis polynomials.  If coefficients are absent,
    the function falls back to the Hough line whose peak angle is closest to the
    selected ``alpha``.  Pass ``alpha_override`` to rebuild the MATLAB-style
    extrapolated fascicle line with an explicit angle.
    """

    fas_coef = np.asarray(entry.get("fas_coef", []), dtype=np.float64).reshape(-1)
    super_coef = np.asarray(entry.get("super_coef", []), dtype=np.float64).reshape(-1)
    deep_coef = np.asarray(entry.get("deep_coef", []), dtype=np.float64).reshape(-1)

    if alpha_override is not None and super_coef.size >= 2 and deep_coef.size >= 2:
        image_shape = np.asarray(entry.get("image_shape", []), dtype=np.float64).reshape(-1)
        if image_shape.size >= 2:
            width = int(image_shape[1])
        else:
            x_lines = np.asarray(entry.get("x", []), dtype=np.float64)
            width = int(np.nanmax(x_lines)) if x_lines.size else 0
        if width > 0:
            return fascicle_segment_from_aponeuroses_and_alpha(
                super_coef,
                deep_coef,
                float(alpha_override),
                width,
                super_coef_linear_1b=entry.get("super_coef_linear", super_coef),
                deep_coef_linear_1b=entry.get("deep_coef_linear", deep_coef),
            )

    if fas_coef.size >= 2 and super_coef.size >= 2 and deep_coef.size >= 2:
        if np.all(np.isfinite(fas_coef[:2])) and np.all(np.isfinite(super_coef[:2])) and np.all(
            np.isfinite(deep_coef[:2])
        ):
            denom_super = fas_coef[0] - super_coef[0]
            denom_deep = fas_coef[0] - deep_coef[0]
            if abs(denom_super) > 1e-12 and abs(denom_deep) > 1e-12:
                x_super = (super_coef[1] - fas_coef[1]) / denom_super
                y_super = np.polyval(fas_coef, x_super)
                x_deep = (deep_coef[1] - fas_coef[1]) / denom_deep
                y_deep = np.polyval(fas_coef, x_deep)
                return np.asarray([x_super, y_super, x_deep, y_deep], dtype=np.float64)

    alpha = float(np.asarray(entry.get("alpha", np.nan), dtype=np.float64).reshape(-1)[0])
    alphas = np.asarray(entry.get("alphas", []), dtype=np.float64).reshape(-1)
    x_lines = np.asarray(entry.get("x", []), dtype=np.float64)
    y_lines = np.asarray(entry.get("y", []), dtype=np.float64)
    if alphas.size and x_lines.ndim == 2 and y_lines.ndim == 2 and x_lines.shape[1] >= 2 and y_lines.shape[1] >= 2:
        idx = int(np.nanargmin(np.abs(alphas - alpha)))
        return np.asarray([x_lines[idx, 0], y_lines[idx, 0], x_lines[idx, 1], y_lines[idx, 1]], dtype=np.float64)

    return np.full(4, np.nan, dtype=np.float64)


def compact_timtrack_geofeature(entry: Mapping[str, Any], *, max_peaks: int = 10) -> dict:
    """Keep only lightweight geofeature fields needed by parity/KLT gates."""

    out: dict[str, Any] = {}
    for key in [
        "frame",
        "alpha",
        "phi",
        "faslen",
        "betha",
        "gamma",
        "thickness",
        "brightness",
        "extrapolated_fraction",
        "apo_x",
        "super_pos",
        "deep_pos",
        "super_coef",
        "deep_coef",
        "super_coef_linear",
        "deep_coef_linear",
        "super_vec_1b",
        "deep_vec_1b",
        "fas_coef",
        "Emask_radius",
        "mean_depths",
        "image_shape",
        "hough_peak_source",
        "hough_localmax_fallback_used",
        "hough_localmax_fallback_mass_below_10deg",
        "hough_localmax_fallback_gap_to_lower_deg",
        "hough_baseline_alpha_deg",
    ]:
        if key in entry:
            out[key] = entry[key]

    for key in ["alphas", "ws", "weights"]:
        values = np.asarray(entry.get(key, []), dtype=np.float64).reshape(-1)
        padded = np.full(int(max_peaks), np.nan, dtype=np.float64)
        n = min(int(max_peaks), len(values))
        if n:
            padded[:n] = values[:n]
        out[key] = padded

    for key in ["x", "y"]:
        values = np.asarray(entry.get(key, []), dtype=np.float64)
        padded = np.full((int(max_peaks), 2), np.nan, dtype=np.float64)
        if values.ndim == 2 and values.shape[1] >= 2:
            n = min(int(max_peaks), values.shape[0])
            padded[:n] = values[:n, :2]
        out[key] = padded

    return out


def run_timtrack_geofeatures_from_video(
    video_path: str,
    parms: Mapping[str, Any],
    *,
    limit: Optional[int] = None,
    subtraction_mode: str = "matlab_literal",
    emask_mode: str = "matlab",
    hough_localmax_fallback: bool = False,
    hough_fallback_min_mass_below_10deg: float = 0.25,
    hough_fallback_min_gap_to_lower_deg: float = 4.0,
    keep_debug: bool = False,
    progress_every: Optional[int] = None,
) -> list[dict]:
    """Generate TimTrack-like geofeatures from each frame of a video."""

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(video_path)
    entries: list[dict] = []
    frame_idx = 0
    while limit is None or frame_idx < int(limit):
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame.copy()
        entry = detect_timtrack_geofeature_from_image(
            gray,
            parms,
            subtraction_mode=subtraction_mode,
            emask_mode=emask_mode,
            hough_localmax_fallback=hough_localmax_fallback,
            hough_fallback_min_mass_below_10deg=hough_fallback_min_mass_below_10deg,
            hough_fallback_min_gap_to_lower_deg=hough_fallback_min_gap_to_lower_deg,
        )
        entry["frame"] = frame_idx
        entries.append(entry if keep_debug else compact_timtrack_geofeature(entry))
        frame_idx += 1
        if progress_every and (frame_idx % int(progress_every) == 0):
            print(f"TimTrack image geofeatures processed {frame_idx}")
    cap.release()
    return entries

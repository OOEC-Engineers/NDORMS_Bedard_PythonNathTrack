"""Autonomous first-frame fascicle seed selection for the strict Python gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
from scipy.ndimage import binary_dilation

from .geometry import line_angles_batch, line_lengths_batch, normalize_angle
from .matlab_timtrack import (
    fascicle_segment_from_aponeuroses_and_alpha,
    fascicle_segment_from_geofeature,
)


@dataclass(frozen=True)
class FascicleSeedScoringConfig:
    """Weights and priors used by the autonomous seed selector."""

    angle_min_deg: float = 14.0
    angle_max_deg: float = 24.0
    angle_step_deg: float = 0.1
    top_peak_limit: int = 10
    min_cluster_frame_coverage: int = 8
    mask_dilation_iterations: int = 3
    weight_mask_support: float = 0.35
    weight_raw_mask_support: float = 0.18
    weight_hough: float = 0.05
    weight_lateral_span: float = 0.12
    weight_phi: float = 0.12
    weight_pennation: float = 0.08
    weight_inside_muscle: float = 0.08
    weight_boundary: float = 0.01
    hough_branch_score_margin: float = 0.02
    hough_branch_alpha_tolerance_deg: float = 3.0


def normalized_segment_angle(segment: np.ndarray) -> float:
    """Return the normalized angle of one ``[x1, y1, x2, y2]`` segment."""

    angle = line_angles_batch(np.asarray(segment, dtype=float).reshape(1, 4), degrees=True)[0]
    return float(normalize_angle(angle, degrees=True))


def sample_segment_points(segment: np.ndarray, n_samples: int = 500) -> tuple[np.ndarray, np.ndarray]:
    """Sample one-based points along a flat segment."""

    x1, y1, x2, y2 = np.asarray(segment, dtype=float).reshape(4)
    t = np.linspace(0.0, 1.0, int(n_samples))
    return x1 + (x2 - x1) * t, y1 + (y2 - y1) * t


def _soft_range_score(value: float, low: float, high: float, scale: float = 10.0) -> float:
    if not np.isfinite(value):
        return 0.0
    if low <= value <= high:
        return 1.0
    return float(np.exp(-min(abs(value - low), abs(value - high)) / scale))


def _entry_value(entry: Mapping, key: str, default=None):
    if key in entry:
        return entry[key]
    return default


def _hough_score_for_alpha(entry: Mapping, alpha_deg: float) -> float:
    hough_result = _entry_value(entry, "hough_result", {}) or {}
    h_by_angle = np.asarray(_entry_value(hough_result, "h_by_angle", []), dtype=float).reshape(-1)
    gamma = np.asarray(_entry_value(hough_result, "gamma", []), dtype=float).reshape(-1)
    if h_by_angle.size == 0 or gamma.size == 0:
        return 0.0
    hmax = float(np.nanmax(h_by_angle))
    if not np.isfinite(hmax) or hmax <= 0:
        return 0.0
    idx = int(np.nanargmin(np.abs(gamma - float(alpha_deg))))
    return float(h_by_angle[idx] / hmax)


def score_fascicle_seed_candidate(
    entry: Mapping,
    segment: np.ndarray,
    alpha_deg: float,
    *,
    mm_per_px: float | None = None,
    frame_shape: tuple[int, int],
    config: FascicleSeedScoringConfig | None = None,
) -> dict[str, float]:
    """Score one autonomous fascicle seed candidate using depth-independent evidence."""

    cfg = config or FascicleSeedScoringConfig()
    height, width = map(int, frame_shape)
    mask = np.asarray(entry["fascicle_masked"], dtype=bool)
    dilated = binary_dilation(mask, iterations=int(cfg.mask_dilation_iterations))
    length_px = float(line_lengths_batch(np.asarray(segment, dtype=float).reshape(1, 4))[0])
    try:
        mm_scale = float(mm_per_px) if mm_per_px is not None else np.nan
    except (TypeError, ValueError):
        mm_scale = np.nan
    length_mm = float(length_px * mm_scale) if np.isfinite(mm_scale) and mm_scale > 0 else float("nan")

    xs, ys = sample_segment_points(segment)
    xi = np.rint(xs).astype(int) - 1
    yi = np.rint(ys).astype(int) - 1
    in_bounds = (xi >= 0) & (xi < width) & (yi >= 0) & (yi < height)
    visible_fraction = float(np.mean(in_bounds)) if len(in_bounds) else 0.0
    hough_score = _hough_score_for_alpha(entry, alpha_deg)
    if int(np.sum(in_bounds)) < 20:
        return {
            "score": 0.0,
            "raw_mask_support": 0.0,
            "mask_support_score": 0.0,
            "inside_muscle_score": 0.0,
            "visible_fraction": visible_fraction,
            "hough_score": hough_score,
            "length_score": 0.0,
            "lateral_span_score": 0.0,
            "length_px": length_px,
            "length_mm": length_mm,
            "phi_score": 0.0,
            "pennation_score": 0.0,
            "boundary_score": 0.0,
        }

    raw_support = float(mask[yi[in_bounds], xi[in_bounds]].mean())
    mask_support = float(dilated[yi[in_bounds], xi[in_bounds]].mean())

    x_1b = xi[in_bounds].astype(float) + 1.0
    y_1b = yi[in_bounds].astype(float) + 1.0
    lateral_span_score = float(np.clip((float(np.nanmax(x_1b) - np.nanmin(x_1b)) + 1.0) / float(width), 0.0, 1.0))
    super_y = np.polyval(np.asarray(entry["super_coef"], dtype=float), x_1b)
    deep_y = np.polyval(np.asarray(entry["deep_coef"], dtype=float), x_1b)
    inside_muscle = float(
        ((y_1b >= np.minimum(super_y, deep_y) - 2.0) & (y_1b <= np.maximum(super_y, deep_y) + 2.0)).mean()
    )

    phi = float(alpha_deg - float(entry["betha"]))
    pennation_deep = float(alpha_deg - float(entry["gamma"]))

    # A physical-length prior makes the selected fascicle depend on image depth.
    # Keep a neutral diagnostic score here and let mask/Hough/geometry evidence decide.
    length_score = 1.0
    phi_score = _soft_range_score(abs(phi), 10.0, 45.0)
    pennation_score = _soft_range_score(abs(pennation_deep), 5.0, 45.0)

    x1, y1, x2, y2 = np.asarray(segment, dtype=float).reshape(4)
    outside_px = max(0.0, -min(x1, x2), max(x1, x2) - width, -min(y1, y2), max(y1, y2) - height)
    boundary_score = float(np.exp(-outside_px / (0.30 * width)))

    score = (
        cfg.weight_mask_support * mask_support
        + cfg.weight_raw_mask_support * raw_support
        + cfg.weight_hough * hough_score
        + cfg.weight_lateral_span * lateral_span_score
        + cfg.weight_phi * phi_score
        + cfg.weight_pennation * pennation_score
        + cfg.weight_inside_muscle * inside_muscle
        + cfg.weight_boundary * boundary_score
    )

    return {
        "score": float(score),
        "raw_mask_support": raw_support,
        "mask_support_score": mask_support,
        "inside_muscle_score": inside_muscle,
        "visible_fraction": visible_fraction,
        "hough_score": hough_score,
        "length_score": length_score,
        "lateral_span_score": lateral_span_score,
        "phi_score": phi_score,
        "pennation_score": pennation_score,
        "boundary_score": boundary_score,
        "length_px": length_px,
        "length_mm": length_mm,
        "phi_deg": phi,
        "pennation_deep_deg": pennation_deep,
        "outside_px": float(outside_px),
    }


def extract_fascicle_seed_candidates(
    entries: Sequence[Mapping],
    frames: Sequence[np.ndarray],
    *,
    mm_per_px: float | None = None,
    config: FascicleSeedScoringConfig | None = None,
    angle_grid: np.ndarray | None = None,
) -> pd.DataFrame:
    """Extract and score Python-only seed candidates from the first frames."""

    cfg = config or FascicleSeedScoringConfig()
    if angle_grid is None:
        angle_grid = np.arange(cfg.angle_min_deg, cfg.angle_max_deg + 1e-12, cfg.angle_step_deg)

    rows: list[dict] = []
    for entry, frame in zip(entries, frames):
        frame_idx = int(_entry_value(entry, "frame", len(rows)))
        peak_alphas = np.asarray(_entry_value(entry, "alphas", []), dtype=float).reshape(-1)[: cfg.top_peak_limit]
        peak_weights = np.asarray(_entry_value(entry, "weights", _entry_value(entry, "ws", [])), dtype=float).reshape(-1)[
            : cfg.top_peak_limit
        ]
        alpha_pool = np.unique(np.round(np.r_[angle_grid, peak_alphas[np.isfinite(peak_alphas)]], 3))
        # ``angle_grid`` already honours the configured seed-only range, but
        # raw Hough peaks are appended so they can compete with grid
        # candidates.  Apply the same range to the combined pool; otherwise a
        # Hough peak outside ``--seed-angle-range`` can bypass the advertised
        # initialization constraint and be selected.
        alpha_pool = alpha_pool[
            (alpha_pool >= float(cfg.angle_min_deg) - 1e-9)
            & (alpha_pool <= float(cfg.angle_max_deg) + 1e-9)
        ]
        max_peak_weight = float(np.nanmax(peak_weights)) if peak_weights.size and np.isfinite(np.nanmax(peak_weights)) else np.nan

        for alpha in alpha_pool:
            segment = fascicle_segment_from_aponeuroses_and_alpha(
                entry["super_coef"],
                entry["deep_coef"],
                float(alpha),
                int(frame.shape[1]),
                super_coef_linear_1b=entry["super_coef_linear"],
                deep_coef_linear_1b=entry["deep_coef_linear"],
            )
            if not np.all(np.isfinite(segment)):
                continue

            scores = score_fascicle_seed_candidate(
                entry,
                segment,
                float(alpha),
                mm_per_px=mm_per_px,
                frame_shape=frame.shape[:2],
                config=cfg,
            )
            nearest_peak_idx = int(np.nanargmin(np.abs(peak_alphas - alpha))) if peak_alphas.size else -1
            nearest_peak_delta = float(abs(peak_alphas[nearest_peak_idx] - alpha)) if nearest_peak_idx >= 0 else np.nan
            is_hough_peak = bool(np.isfinite(nearest_peak_delta) and nearest_peak_delta <= 0.051)
            peak_weight = float(peak_weights[nearest_peak_idx]) if is_hough_peak and nearest_peak_idx < len(peak_weights) else np.nan

            x1, y1, x2, y2 = np.asarray(segment, dtype=float).reshape(4)
            rows.append(
                {
                    "frame": frame_idx,
                    "candidate_source": "hough_peak" if is_hough_peak else "grid_scan",
                    "alpha_deg": float(alpha),
                    "segment_angle_deg": normalized_segment_angle(segment),
                    "x_sup": float(x1),
                    "y_sup": float(y1),
                    "x_deep": float(x2),
                    "y_deep": float(y2),
                    "x_mid": float((x1 + x2) / 2.0),
                    "y_mid": float((y1 + y2) / 2.0),
                    "peak_weight": peak_weight,
                    "peak_weight_norm": (
                        float(peak_weight / max_peak_weight)
                        if np.isfinite(peak_weight) and np.isfinite(max_peak_weight) and max_peak_weight > 0
                        else np.nan
                    ),
                    **scores,
                }
            )

    return pd.DataFrame(rows).sort_values(["frame", "score"], ascending=[True, False]).reset_index(drop=True)


def cluster_seed_candidates(
    candidates: pd.DataFrame,
    *,
    min_frame_coverage: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster candidates by angle, position, and pixel length and score stability."""

    if candidates.empty:
        return candidates.copy(), pd.DataFrame()

    min_coverage = int(min_frame_coverage or FascicleSeedScoringConfig().min_cluster_frame_coverage)
    clustered = candidates.copy()
    if "length_px" not in clustered:
        required = {"x_sup", "y_sup", "x_deep", "y_deep"}
        if required.issubset(clustered.columns):
            clustered["length_px"] = np.hypot(
                clustered["x_deep"] - clustered["x_sup"],
                clustered["y_deep"] - clustered["y_sup"],
            )
        else:
            clustered["length_px"] = np.nan
    if "length_mm" not in clustered:
        clustered["length_mm"] = np.nan
    clustered["alpha_bin"] = np.round(clustered["alpha_deg"] / 0.25) * 0.25
    clustered["xmid_bin"] = np.round(clustered["x_mid"] / 50.0) * 50.0
    clustered["length_bin_px"] = np.round(clustered["length_px"] / 50.0) * 50.0
    clustered["length_bin"] = clustered["length_bin_px"]
    clustered["cluster_id"] = (
        clustered["alpha_bin"].map(lambda x: f"a{x:.2f}")
        + "_x"
        + clustered["xmid_bin"].map(lambda x: f"{x:.0f}")
        + "_lp"
        + clustered["length_bin_px"].map(lambda x: f"{x:.0f}")
    )

    rows: list[dict] = []
    for cluster_id, group in clustered.groupby("cluster_id"):
        coverage = int(group["frame"].nunique())
        if coverage < min_coverage:
            continue
        per_frame_best = group.sort_values("score", ascending=False).groupby("frame", as_index=False).head(1)
        length_std_px = float(per_frame_best["length_px"].std(ddof=0))
        if not np.isfinite(length_std_px):
            length_std_px = 0.0
        cluster_score = (
            float(per_frame_best["score"].mean())
            + 0.04 * coverage
            - 0.02 * float(per_frame_best["alpha_deg"].std(ddof=0))
            - 0.000125 * length_std_px
        )
        rows.append(
            {
                "cluster_id": cluster_id,
                "frame_coverage": coverage,
                "n_candidates": int(len(group)),
                "cluster_score": cluster_score,
                "mean_score": float(per_frame_best["score"].mean()),
                "median_alpha_deg": float(per_frame_best["alpha_deg"].median()),
                "alpha_std_deg": float(per_frame_best["alpha_deg"].std(ddof=0)),
                "median_length_px": float(per_frame_best["length_px"].median()),
                "length_std_px": length_std_px,
                "median_length_mm": float(per_frame_best["length_mm"].median()),
                "length_std_mm": float(per_frame_best["length_mm"].std(ddof=0)),
                "median_x_mid": float(per_frame_best["x_mid"].median()),
                "x_mid_std": float(per_frame_best["x_mid"].std(ddof=0)),
                "mean_mask_support": float(per_frame_best["mask_support_score"].mean()),
                "mean_hough_score": float(per_frame_best["hough_score"].mean()),
                "mean_lateral_span_score": float(per_frame_best["lateral_span_score"].mean()),
                "hough_peak_fraction": float((per_frame_best["candidate_source"] == "hough_peak").mean()),
            }
        )

    clusters = pd.DataFrame(rows).sort_values("cluster_score", ascending=False).reset_index(drop=True)
    return clustered, clusters


def choose_stable_seed_cluster(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    *,
    config: FascicleSeedScoringConfig | None = None,
) -> dict:
    """Choose the seed cluster, preferring a near-tied stable Hough branch."""

    if clusters.empty:
        raise RuntimeError("No stable fascicle seed candidate cluster found.")

    cfg = config or FascicleSeedScoringConfig()
    selected_cluster = clusters.iloc[0].to_dict()
    hough_candidates = candidates[candidates["candidate_source"] == "hough_peak"]
    if hough_candidates.empty:
        return selected_cluster

    per_frame_hough = hough_candidates.sort_values("score", ascending=False).groupby("frame", as_index=False).head(1)
    if int(per_frame_hough["frame"].nunique()) < int(cfg.min_cluster_frame_coverage):
        return selected_cluster

    hough_alpha = float(per_frame_hough["alpha_deg"].median())
    if not np.isfinite(hough_alpha):
        return selected_cluster

    best_score = float(selected_cluster["cluster_score"])
    close_clusters = clusters[
        clusters["cluster_score"] >= best_score - float(cfg.hough_branch_score_margin)
    ].copy()
    close_clusters["hough_alpha_delta"] = np.abs(close_clusters["median_alpha_deg"] - hough_alpha)
    hough_branch = close_clusters[
        close_clusters["hough_alpha_delta"] <= float(cfg.hough_branch_alpha_tolerance_deg)
    ]
    if hough_branch.empty:
        return selected_cluster

    return hough_branch.sort_values(
        ["cluster_score", "mean_hough_score", "mean_mask_support"],
        ascending=[False, False, False],
    ).iloc[0].to_dict()


def select_autonomous_fascicle_seed(
    candidates: pd.DataFrame,
    clusters: pd.DataFrame,
    first_entry: Mapping,
    width_px: int,
) -> dict:
    """Select the median seed from the most stable Python-only cluster."""

    if clusters.empty:
        raise RuntimeError("No stable fascicle seed candidate cluster found.")
    selected_cluster = choose_stable_seed_cluster(candidates, clusters)
    cluster_members = candidates[candidates["cluster_id"] == selected_cluster["cluster_id"]]
    per_frame_best = cluster_members.sort_values("score", ascending=False).groupby("frame", as_index=False).head(1)
    selected_alpha = float(per_frame_best["alpha_deg"].median())
    selected_seed = fascicle_segment_from_aponeuroses_and_alpha(
        first_entry["super_coef"],
        first_entry["deep_coef"],
        selected_alpha,
        int(width_px),
        super_coef_linear_1b=first_entry["super_coef_linear"],
        deep_coef_linear_1b=first_entry["deep_coef_linear"],
    )
    return {
        "selected_cluster": selected_cluster,
        "selected_alpha_deg": selected_alpha,
        "selected_seed_segment": np.asarray(selected_seed, dtype=float),
        "cluster_members": cluster_members,
        "per_frame_best": per_frame_best,
    }


def select_matlab_first_frame_seed(first_entry: Mapping) -> dict:
    """Use MATLAB's first-frame weighted-median Hough angle and centred line.

    The baseline Hough angle is preferred so the optional Python local-maximum
    fallback cannot alter this MATLAB-initialization sensitivity mode.
    """

    baseline = float(np.asarray(first_entry.get("hough_baseline_alpha_deg", np.nan)).reshape(-1)[0])
    detected = float(np.asarray(first_entry.get("alpha", np.nan)).reshape(-1)[0])
    selected_alpha = baseline if np.isfinite(baseline) else detected
    if not np.isfinite(selected_alpha):
        raise RuntimeError("The first TimTrack frame has no finite weighted-median Hough angle.")

    selected_seed = fascicle_segment_from_geofeature(first_entry, alpha_override=selected_alpha)
    if not np.all(np.isfinite(selected_seed)):
        raise RuntimeError("The first TimTrack frame could not produce a finite MATLAB-centred fascicle line.")

    cluster_id = "matlab_first_frame_weighted_median"
    audit = pd.DataFrame(
        [
            {
                "frame": 0,
                "candidate_source": cluster_id,
                "alpha_deg": selected_alpha,
                "segment_angle_deg": normalized_segment_angle(selected_seed),
                "x_sup": float(selected_seed[0]),
                "y_sup": float(selected_seed[1]),
                "x_deep": float(selected_seed[2]),
                "y_deep": float(selected_seed[3]),
                "cluster_id": cluster_id,
            }
        ]
    )
    return {
        "selected_cluster": {
            "cluster_id": cluster_id,
            "frame_coverage": 1,
            "median_alpha_deg": selected_alpha,
            "selection_rule": "first-frame baseline weighted-median Hough angle plus MATLAB-centred intercept",
        },
        "selected_alpha_deg": selected_alpha,
        "selected_seed_segment": np.asarray(selected_seed, dtype=float),
        "cluster_members": audit.copy(),
        "per_frame_best": audit,
    }

import zipfile

import numpy as np
import pytest

from scripts.run_strict_ultratimtrack_video import build_arg_parser as build_runner_arg_parser
from scripts.strict_ultratimtrack_gui import (
    _has_metric_values,
    _match_speckle_point_with_fb,
    _read_metrics_csv,
    _runner_namespace,
    _speckle_config_from_box,
    _write_simple_xlsx,
)


def test_gui_runner_namespace_tracks_all_runner_defaults():
    runner_defaults = vars(build_runner_arg_parser().parse_args([]))

    gui_args = _runner_namespace(seed_frames=17)

    assert set(vars(gui_args)) == set(runner_defaults)
    assert gui_args.seed_frames == 17
    assert gui_args.seed_angle_range is None


def test_gui_runner_namespace_rejects_unknown_fields():
    with pytest.raises(TypeError, match="Unknown strict runner argument"):
        _runner_namespace(no_longer_a_runner_option=True)


def test_read_metrics_csv_loads_fixed_kalman_comparison_columns(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text(
        "\n".join(
            [
                "Frame,Time,FL,PEN,ANG,fixed_FL_mm,fixed_PEN_deg,fixed_ANG_deg",
                "0,0.0,22.0,45.0,45.5,23.0,44.0,44.5",
                "1,0.1,24.0,40.0,40.5,25.0,39.0,39.5",
            ]
        )
        + "\n"
    )

    metrics = _read_metrics_csv(csv_path)

    assert metrics["FL"] == [22.0, 24.0]
    assert metrics["PEN"] == [45.0, 40.0]
    assert metrics["ANG"] == [45.5, 40.5]
    assert metrics["FixedFL"] == [23.0, 25.0]
    assert metrics["FixedPEN"] == [44.0, 39.0]
    assert metrics["FixedANG"] == [44.5, 39.5]
    assert _has_metric_values(metrics["FixedFL"])


def test_read_metrics_csv_keeps_fixed_columns_empty_when_not_comparing(tmp_path):
    csv_path = tmp_path / "metrics.csv"
    csv_path.write_text("Frame,Time,FL,PEN,ANG\n0,0.0,22.0,45.0,45.5\n")

    metrics = _read_metrics_csv(csv_path)

    assert metrics["FL"] == [22.0]
    assert metrics["FixedFL"] == []
    assert not _has_metric_values(metrics["FixedFL"])


def test_speckle_patch_matcher_recovers_known_shift():
    rng = np.random.default_rng(123)
    frame0 = rng.normal(100.0, 25.0, (80, 80)).astype(np.float32)
    frame1 = np.zeros_like(frame0)
    frame1[2:, 3:] = frame0[:-2, :-3]
    cfg = _speckle_config_from_box(21)

    match = _match_speckle_point_with_fb(frame0, frame1, np.asarray([40.0, 40.0]), cfg)

    assert match["ok"]
    assert np.allclose(match["point"], [43.0, 42.0])
    assert match["zncc"] > 0.99


def test_speckle_config_rejects_even_box_size():
    with pytest.raises(ValueError, match="odd"):
        _speckle_config_from_box(40)


def test_write_simple_xlsx_creates_workbook(tmp_path):
    output = tmp_path / "tracking.xlsx"
    _write_simple_xlsx(
        output,
        {
            "tracking": [{"frame": 0, "point_id": "center", "valid": True, "d_parallel_mm": 1.25}],
            "summary": [{"mean_zncc": 0.98}],
        },
    )

    with zipfile.ZipFile(output) as zf:
        names = set(zf.namelist())
        workbook = zf.read("xl/workbook.xml").decode("utf-8")

    assert "[Content_Types].xml" in names
    assert "xl/worksheets/sheet1.xml" in names
    assert 'name="tracking"' in workbook

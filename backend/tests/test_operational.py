"""Tests for operational features: georeferencing, confidence estimation, and artifact packaging."""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np
import pytest

from app.services.georeferencing import align_reconstruction_to_gps
from app.services.confidence import compute_confidence_map


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_sfm_and_gps_data(tmp_path: Path):
    """Create synthetic camera poses and matching GPS telemetry."""
    ws = tmp_path / "op_ws"
    ws.mkdir()

    poses = []
    telemetry = []

    # Reference origin
    lat0, lon0, alt0 = 28.6139, 77.2090, 215.0

    for i in range(10):
        # Camera moving along x-axis in SfM
        poses.append({
            "image_name": f"frame_{i:04d}.jpg",
            "position": [float(i * 1.5), 0.0, 5.0],
            "rotation_matrix": np.eye(3).tolist(),
        })

        # Telemetry: drone flying East along longitude (~1.5m intervals scaled)
        # 1 deg lon ~ 111320m * cos(28.6139) ~ 97700m
        # 3m ~ 3 / 97700 deg
        telemetry.append({
            "timestamp_s": float(i * 0.5),
            "latitude": lat0 + (i * 0.00001),
            "longitude": lon0 + (i * 0.00003),
            "altitude_m": alt0 + (i * 0.2),
        })

    poses_file = ws / "camera_poses.json"
    poses_file.write_text(json.dumps(poses, indent=2))

    return {
        "poses_json": str(poses_file),
        "telemetry": telemetry,
        "output_dir": str(ws),
    }


# ── Georeferencing Tests ─────────────────────────────────────────────


def test_align_reconstruction_with_gps(synthetic_sfm_and_gps_data: dict):
    res = align_reconstruction_to_gps(
        poses_json_path=synthetic_sfm_and_gps_data["poses_json"],
        telemetry_records=synthetic_sfm_and_gps_data["telemetry"],
        output_dir=synthetic_sfm_and_gps_data["output_dir"],
    )

    assert res["success"] is True
    assert res["stats"]["status"] == "aligned"
    assert res["stats"]["scale_factor_m_per_unit"] > 0
    assert res["stats"]["matched_fixes"] >= 3
    assert res["stats"]["alignment_rmse_m"] is not None

    aligned_poses_p = Path(res["aligned_poses_json"])
    assert aligned_poses_p.exists()
    aligned_data = json.loads(aligned_poses_p.read_text())
    assert len(aligned_data) == 10
    assert "metric_position_enu_m" in aligned_data[0]

    geojson_p = Path(res["geojson_path"])
    assert geojson_p.exists()
    geojson_data = json.loads(geojson_p.read_text())
    assert geojson_data["type"] == "FeatureCollection"
    assert len(geojson_data["features"]) > 0


def test_align_reconstruction_without_gps(synthetic_sfm_and_gps_data: dict):
    res = align_reconstruction_to_gps(
        poses_json_path=synthetic_sfm_and_gps_data["poses_json"],
        telemetry_records=None,
        output_dir=synthetic_sfm_and_gps_data["output_dir"],
    )

    assert res["success"] is True
    assert res["stats"]["status"] == "uncalibrated"
    assert res["stats"]["scale_factor_m_per_unit"] == 1.0


# ── Confidence Mapping Tests ─────────────────────────────────────────


def test_compute_confidence_map_high_quality(tmp_path: Path):
    res = compute_confidence_map(
        points_count=15000,
        registered_images=30,
        mean_reprojection_error=0.65,
        mean_flow_consistency=0.92,
        has_gps=True,
        output_dir=str(tmp_path),
    )

    assert res["success"] is True
    summary = res["summary"]
    assert summary["overall_tier"] == "high"
    assert summary["composite_score"] > 0.75
    assert summary["distribution"]["high"] > 60.0
    assert sum(summary["distribution"].values()) == pytest.approx(100.0, abs=0.5)

    assert res["report_path"] is not None
    assert Path(res["report_path"]).exists()


def test_compute_confidence_map_low_quality(tmp_path: Path):
    res = compute_confidence_map(
        points_count=120,
        registered_images=3,
        mean_reprojection_error=3.5,
        mean_flow_consistency=0.4,
        has_gps=False,
        output_dir=str(tmp_path),
    )

    assert res["success"] is True
    summary = res["summary"]
    assert summary["overall_tier"] in ("medium", "low")
    assert summary["composite_score"] < 0.60

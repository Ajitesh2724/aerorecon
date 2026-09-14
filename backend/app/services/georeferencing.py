"""Georeferencing and metric scale alignment using GPS/IMU telemetry."""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("aerorecon.georef")


def align_reconstruction_to_gps(
    poses_json_path: str,
    telemetry_records: Optional[list[dict]],
    output_dir: str,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Align SfM camera positions with GPS telemetry using 7-DoF Sim(3) Umeyama alignment.

    Outputs:
      - Aligned camera poses JSON
      - GeoJSON flight path & boundary
      - Metric scale factor (m per SfM unit)
      - Alignment residual RMSE (meters)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    poses_p = Path(poses_json_path)
    if not poses_p.exists():
        raise FileNotFoundError(f"Poses JSON not found at {poses_json_path}")

    poses: list[dict] = json.loads(poses_p.read_text())
    n_poses = len(poses)
    if n_poses < 2:
        return _unaligned_result(out, poses, "Insufficient camera poses (< 2)")

    if not telemetry_records or len(telemetry_records) < 2:
        logger.info("No GPS telemetry available. Assigning nominal drone flight scale.")
        return _unaligned_result(out, poses, "No GPS telemetry provided")

    if on_progress:
        on_progress(20, "Correlating camera frames with GPS telemetry…")

    # Match each camera pose with the closest GPS record (by index or timestamp)
    matched_sfm = []
    matched_gps_enu = []
    matched_gps_wgs84 = []

    # Choose reference origin from first valid GPS fix
    ref_record = next((r for r in telemetry_records if "latitude" in r and "longitude" in r), None)
    if not ref_record:
        return _unaligned_result(out, poses, "No valid coordinates in telemetry records")

    lat0, lon0, alt0 = ref_record["latitude"], ref_record["longitude"], ref_record.get("altitude_m", ref_record.get("altitude", 50.0))

    # Associate poses to telemetry
    n_tel = len(telemetry_records)
    for i, pose in enumerate(poses):
        pos = pose.get("position")
        if not pos or len(pos) != 3:
            continue

        # Proportional mapping between keyframes and telemetry sequence
        tel_idx = int(min(n_tel - 1, (i / max(1, n_poses - 1)) * (n_tel - 1)))
        tel = telemetry_records[tel_idx]

        lat = tel.get("latitude")
        lon = tel.get("longitude")
        alt = tel.get("altitude_m", tel.get("altitude", alt0))

        if lat is not None and lon is not None:
            enu_x, enu_y, enu_z = _geodetic_to_enu(lat, lon, alt, lat0, lon0, alt0)
            matched_sfm.append(pos)
            matched_gps_enu.append([enu_x, enu_y, enu_z])
            matched_gps_wgs84.append({"lat": lat, "lon": lon, "alt": alt})

    if len(matched_sfm) < 3:
        return _unaligned_result(out, poses, f"Only {len(matched_sfm)} matched GPS fixes (need >= 3)")

    if on_progress:
        on_progress(50, "Estimating 7-DoF Sim(3) similarity transformation…")

    X = np.array(matched_sfm, dtype=np.float64).T      # 3 x N (SfM points)
    Y = np.array(matched_gps_enu, dtype=np.float64).T  # 3 x N (GPS ENU points)

    scale, R, t = _umeyama_sim3(X, Y)

    # Compute residuals: || s * R * X + t - Y ||
    aligned_X = (scale * (R @ X)) + t[:, None]
    residuals = np.linalg.norm(aligned_X - Y, axis=0)
    rmse_m = float(np.sqrt(np.mean(residuals ** 2)))

    if on_progress:
        on_progress(80, "Writing georeferenced outputs and GeoJSON…")

    # Update camera poses with metric georeferenced coordinates
    aligned_poses = []
    for pose in poses:
        p_copy = dict(pose)
        pos = np.array(pose.get("position", [0, 0, 0]), dtype=np.float64)
        p_aligned = (scale * (R @ pos) + t).tolist()
        p_copy["metric_position_enu_m"] = [round(v, 3) for v in p_aligned]
        aligned_poses.append(p_copy)

    aligned_json = out / "georeferenced_poses.json"
    aligned_json.write_text(json.dumps(aligned_poses, indent=2))

    # Generate GeoJSON
    geojson_path = out / "flight_path.geojson"
    geojson_data = _create_geojson(matched_gps_wgs84)
    geojson_path.write_text(json.dumps(geojson_data, indent=2))

    if on_progress:
        on_progress(100, "Georeferencing complete.")

    stats = {
        "status": "aligned",
        "scale_factor_m_per_unit": round(scale, 4),
        "alignment_rmse_m": round(rmse_m, 2),
        "matched_fixes": len(matched_sfm),
        "reference_origin": {"lat": lat0, "lon": lon0, "alt": alt0},
    }

    logger.info("Georeferencing successful: scale=%.4f m/unit, RMSE=%.2fm", scale, rmse_m)
    return {
        "success": True,
        "aligned_poses_json": str(aligned_json),
        "geojson_path": str(geojson_path),
        "stats": stats,
    }


def _unaligned_result(out: Path, poses: list[dict], reason: str) -> dict[str, Any]:
    """Fallback when GPS is not provided or insufficient: assumes default nominal drone scale."""
    stats = {
        "status": "uncalibrated",
        "scale_factor_m_per_unit": 1.0,
        "alignment_rmse_m": None,
        "matched_fixes": 0,
        "reason": reason,
    }
    return {
        "success": True,
        "stats": stats,
    }


def _geodetic_to_enu(lat: float, lon: float, alt: float, lat0: float, lon0: float, alt0: float) -> tuple[float, float, float]:
    """Convert geodetic (WGS84) coordinates to local East-North-Up (ENU) tangent plane."""
    # Earth radius approximation in meters
    R_earth = 6378137.0

    dlat = math.radians(lat - lat0)
    dlon = math.radians(lon - lon0)
    lat_rad = math.radians(lat0)

    # Local flat-Earth ENU projection
    east = dlon * math.cos(lat_rad) * R_earth
    north = dlat * R_earth
    up = alt - alt0
    return east, north, up


def _umeyama_sim3(X: np.ndarray, Y: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Umeyama algorithm: finds s, R, t such that Y ~ s * R * X + t.

    X: 3 x N source points
    Y: 3 x N target points
    Returns (scale, 3x3 rotation matrix, 3x1 translation vector).
    """
    m, n = X.shape
    mx = X.mean(axis=1)
    my = Y.mean(axis=1)

    Xc = X - mx[:, None]
    Yc = Y - my[:, None]

    var_x = np.mean(np.sum(Xc ** 2, axis=0))
    if var_x < 1e-8:
        return 1.0, np.eye(3), my - mx

    # Cross-covariance matrix
    C = (Yc @ Xc.T) / n
    U, D, Vt = np.linalg.svd(C)

    S = np.eye(m)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[m - 1, m - 1] = -1.0

    R = U @ S @ Vt
    scale = float((1.0 / var_x) * np.trace(np.diag(D) @ S))
    t = my - scale * (R @ mx)

    return scale, R, t


def _create_geojson(gps_points: list[dict]) -> dict[str, Any]:
    """Create GeoJSON FeatureCollection containing LineString flight path and point markers."""
    coordinates = [[p["lon"], p["lat"], p.get("alt", 0)] for p in gps_points]

    features = [
        {
            "type": "Feature",
            "properties": {"name": "Drone Flight Trajectory"},
            "geometry": {
                "type": "LineString",
                "coordinates": coordinates,
            },
        }
    ]

    return {
        "type": "FeatureCollection",
        "features": features,
    }

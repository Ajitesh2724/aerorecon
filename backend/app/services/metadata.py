"""Telemetry and camera metadata parsing."""

from __future__ import annotations

import csv
import json
import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("aerorecon.metadata")


# ── GPS / IMU telemetry ──────────────────────────────────────────


def parse_telemetry(file_path: str) -> Optional[list[dict[str, Any]]]:
    """Parse GPS/IMU telemetry from CSV, SRT, or JSON.

    Returns a list of records:
      { timestamp_s, latitude, longitude, altitude_m, yaw, pitch, roll }
    Fields may be None if not available.
    """
    path = Path(file_path)
    if not path.exists():
        logger.warning("Telemetry file not found: %s", file_path)
        return None

    ext = path.suffix.lower()
    try:
        if ext == ".csv":
            return _parse_csv_telemetry(path)
        elif ext == ".srt":
            return _parse_srt_telemetry(path)
        elif ext == ".json":
            return _parse_json_telemetry(path)
        elif ext in (".txt", ".log"):
            return _parse_csv_telemetry(path)  # try CSV-style
        else:
            logger.warning("Unknown telemetry format: %s", ext)
            return None
    except Exception as exc:
        logger.error("Failed to parse telemetry %s: %s", file_path, exc)
        return None


def _parse_csv_telemetry(path: Path) -> list[dict[str, Any]]:
    """Parse comma/tab-separated GPS logs.

    Expected columns (case-insensitive, flexible naming):
      timestamp, latitude/lat, longitude/lon/lng, altitude/alt, yaw, pitch, roll
    """
    records: list[dict[str, Any]] = []
    with open(path, "r", encoding="utf-8-sig") as f:
        # Detect delimiter
        sample = f.read(2048)
        f.seek(0)
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t")
        reader = csv.DictReader(f, dialect=dialect)

        if not reader.fieldnames:
            return records

        # Normalise headers
        col_map: dict[str, str] = {}
        for col in reader.fieldnames:
            lc = col.strip().lower().replace(" ", "_")
            if lc in ("timestamp", "time", "time_s", "timestamp_s"):
                col_map[col] = "timestamp_s"
            elif lc in ("latitude", "lat"):
                col_map[col] = "latitude"
            elif lc in ("longitude", "lon", "lng", "long"):
                col_map[col] = "longitude"
            elif lc in ("altitude", "alt", "altitude_m", "height"):
                col_map[col] = "altitude_m"
            elif lc in ("yaw", "heading"):
                col_map[col] = "yaw"
            elif lc in ("pitch",):
                col_map[col] = "pitch"
            elif lc in ("roll",):
                col_map[col] = "roll"

        for row in reader:
            rec: dict[str, Any] = {}
            for orig, canon in col_map.items():
                val = row.get(orig, "").strip()
                if val:
                    try:
                        rec[canon] = float(val)
                    except ValueError:
                        rec[canon] = val
            if rec:
                records.append(rec)

    logger.info("Parsed %d CSV telemetry records from %s", len(records), path.name)
    return records


def _parse_srt_telemetry(path: Path) -> list[dict[str, Any]]:
    """Parse DJI-style SRT subtitle files with embedded GPS data.

    Example SRT entry:
      1
      00:00:00,000 --> 00:00:01,000
      F/2.8, SS 100, ISO 100, EV 0, GPS (28.5567, 77.1025, 50), ...
    """
    records: list[dict[str, Any]] = []
    text = path.read_text(encoding="utf-8", errors="replace")

    # Find GPS coordinates: GPS (lat, lon, alt) or [latitude: ...] patterns
    gps_pattern = re.compile(
        r"GPS\s*\(\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)",
        re.IGNORECASE,
    )
    time_pattern = re.compile(r"(\d{2}):(\d{2}):(\d{2})[,.](\d{3})")

    blocks = re.split(r"\n\s*\n", text)
    for block in blocks:
        lines = block.strip().split("\n")
        if len(lines) < 2:
            continue

        # Parse timestamp
        ts_match = time_pattern.search(block)
        timestamp_s = None
        if ts_match:
            h, m, s, ms = (int(x) for x in ts_match.groups())
            timestamp_s = h * 3600 + m * 60 + s + ms / 1000.0

        # Parse GPS
        gps_match = gps_pattern.search(block)
        if gps_match:
            lat, lon, alt = (float(x) for x in gps_match.groups())
            records.append({
                "timestamp_s": timestamp_s,
                "latitude": lat,
                "longitude": lon,
                "altitude_m": alt,
            })

    logger.info("Parsed %d SRT telemetry records from %s", len(records), path.name)
    return records


def _parse_json_telemetry(path: Path) -> list[dict[str, Any]]:
    """Parse JSON telemetry (array of objects or {telemetry: [...]})."""
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        # Try common top-level keys
        for key in ("telemetry", "data", "points", "records", "gps"):
            if key in data and isinstance(data[key], list):
                data = data[key]
                break
    if not isinstance(data, list):
        logger.warning("JSON telemetry is not a list.")
        return []

    logger.info("Parsed %d JSON telemetry records from %s", len(data), path.name)
    return data


# ── Camera intrinsics ────────────────────────────────────────────


def parse_camera_intrinsics(raw: str) -> Optional[dict[str, Any]]:
    """Parse camera intrinsics from a JSON string.

    Expected format:
      {"fx": float, "fy": float, "cx": float, "cy": float,
       "k1": float, "k2": float, "p1": float, "p2": float}
    """
    if not raw or not raw.strip():
        return None
    try:
        data = json.loads(raw)
        if not isinstance(data, dict):
            return None
        result: dict[str, Any] = {}
        for key in ("fx", "fy", "cx", "cy", "k1", "k2", "p1", "p2", "k3",
                     "width", "height", "model"):
            if key in data:
                result[key] = data[key]
        if "fx" not in result or "fy" not in result:
            logger.warning("Camera intrinsics missing fx/fy.")
            return None
        logger.info("Parsed camera intrinsics: %s", result)
        return result
    except (json.JSONDecodeError, TypeError) as exc:
        logger.warning("Invalid camera intrinsics JSON: %s", exc)
        return None


def telemetry_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarise telemetry for display in the UI."""
    if not records:
        return {"available": False, "count": 0}

    has_gps = any(r.get("latitude") is not None for r in records)
    has_imu = any(r.get("yaw") is not None for r in records)

    summary: dict[str, Any] = {
        "available": True,
        "count": len(records),
        "has_gps": has_gps,
        "has_imu": has_imu,
    }

    if has_gps:
        lats = [r["latitude"] for r in records if r.get("latitude") is not None]
        lons = [r["longitude"] for r in records if r.get("longitude") is not None]
        alts = [r["altitude_m"] for r in records if r.get("altitude_m") is not None]
        summary["bbox"] = {
            "lat_min": min(lats),
            "lat_max": max(lats),
            "lon_min": min(lons),
            "lon_max": max(lons),
        }
        if alts:
            summary["alt_range_m"] = {"min": min(alts), "max": max(alts)}

    return summary

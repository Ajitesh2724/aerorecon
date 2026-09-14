"""Tests for video preprocessing, keyframe selection, and metadata parsing."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.preprocessing import (
    compute_blur_score,
    compute_motion_score,
    compute_histogram_diff,
    validate_video,
    extract_all_frames,
)
from app.services.keyframes import select_keyframes, summarise_keyframes
from app.services.metadata import (
    parse_camera_intrinsics,
    parse_telemetry,
    telemetry_summary,
)


# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture
def synthetic_video(tmp_path: Path) -> str:
    """Create a short synthetic video for testing (60 frames, 30fps)."""
    video_path = str(tmp_path / "test_video.mp4")
    width, height, fps, n_frames = 320, 240, 30, 60

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    for i in range(n_frames):
        # Create frames with moving gradient (simulates drone motion)
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        offset = int((i / n_frames) * width)
        for c in range(3):
            frame[:, :, c] = np.roll(
                np.tile(np.linspace(50, 200, width, dtype=np.uint8), (height, 1)),
                offset,
                axis=1,
            )
        # Add some structure for blur scoring
        cv2.circle(frame, (width // 2 + offset % 50, height // 2), 30, (255, 255, 255), -1)
        writer.write(frame)

    writer.release()
    return video_path


@pytest.fixture
def sample_frames_meta() -> list[dict]:
    """Simulated frame metadata for keyframe testing."""
    frames = []
    for i in range(200):
        motion = 0.05 + 0.02 * np.sin(i / 10.0)
        blur = 150 + 100 * np.cos(i / 15.0)
        hist = 0.95 if i % 50 != 0 else 0.4  # scene change every 50 frames
        frames.append({
            "index": i,
            "path": f"/tmp/frame_{i:06d}.jpg",
            "blur_score": round(blur, 2),
            "motion_score": round(motion, 4),
            "histogram_diff": round(hist, 4),
            "is_usable": blur >= 100,
        })
    return frames


# ── Preprocessing tests ──────────────────────────────────────────


def test_validate_video_success(synthetic_video: str):
    info = validate_video(synthetic_video)
    assert info.width == 320
    assert info.height == 240
    assert info.fps == 30
    assert info.frame_count == 60
    assert abs(info.duration_s - 2.0) < 0.1


def test_validate_video_missing():
    with pytest.raises(ValueError, match="not found"):
        validate_video("/nonexistent/video.mp4")


def test_blur_score():
    # Sharp image (high-frequency pattern)
    sharp = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    # Blurry image
    blurry = cv2.GaussianBlur(sharp, (21, 21), 10)

    sharp_score = compute_blur_score(sharp)
    blurry_score = compute_blur_score(blurry)
    assert sharp_score > blurry_score, "Sharp image should have higher blur score"


def test_motion_score():
    static = np.zeros((100, 100), dtype=np.uint8)
    moved = np.full((100, 100), 128, dtype=np.uint8)

    score_zero = compute_motion_score(static, static)
    score_high = compute_motion_score(static, moved)

    assert score_zero < score_high, "Motion between different frames should score higher"
    assert score_zero == 0.0


def test_histogram_diff():
    frame_a = np.full((100, 100, 3), [100, 120, 140], dtype=np.uint8)
    frame_b = np.full((100, 100, 3), [100, 120, 140], dtype=np.uint8)
    frame_c = np.full((100, 100, 3), [200, 50, 30], dtype=np.uint8)

    same = compute_histogram_diff(frame_a, frame_b)
    diff = compute_histogram_diff(frame_a, frame_c)
    assert same > diff, "Same frames should have higher correlation"


def test_extract_all_frames(synthetic_video: str, tmp_path: Path):
    output_dir = str(tmp_path / "frames")
    frames = extract_all_frames(synthetic_video, output_dir, max_frames=10)

    assert len(frames) == 10
    assert all("blur_score" in f for f in frames)
    assert all("motion_score" in f for f in frames)
    assert all(Path(f["path"]).exists() for f in frames)


# ── Keyframe tests ────────────────────────────────────────────────


def test_select_keyframes_basic(sample_frames_meta: list[dict]):
    keyframes = select_keyframes(sample_frames_meta, target_count=20)

    assert len(keyframes) <= 25  # target + boundary additions
    assert len(keyframes) >= 15  # reasonable minimum
    assert all(f["is_keyframe"] for f in keyframes)
    # Ensure first/last are included
    indices = [f["index"] for f in keyframes]
    assert 0 in indices, "First frame should be a keyframe"
    assert 199 in indices, "Last frame should be a keyframe"


def test_select_keyframes_few_frames():
    few = [
        {"index": i, "path": f"/f_{i}.jpg", "blur_score": 200, "motion_score": 0.05, "histogram_diff": 0.9, "is_usable": True}
        for i in range(5)
    ]
    result = select_keyframes(few, target_count=20)
    assert len(result) == 5, "Should return all frames when fewer than target"


def test_keyframe_summary(sample_frames_meta: list[dict]):
    keyframes = select_keyframes(sample_frames_meta, target_count=20)
    summary = summarise_keyframes(keyframes)
    assert summary["count"] > 0
    assert "blur_mean" in summary
    assert "selection_reasons" in summary


# ── Metadata tests ────────────────────────────────────────────────


def test_parse_camera_intrinsics_valid():
    raw = '{"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}'
    result = parse_camera_intrinsics(raw)
    assert result is not None
    assert result["fx"] == 1000
    assert result["fy"] == 1000


def test_parse_camera_intrinsics_invalid():
    assert parse_camera_intrinsics("") is None
    assert parse_camera_intrinsics("not json") is None
    assert parse_camera_intrinsics('{"cx": 100}') is None  # missing fx/fy


def test_parse_csv_telemetry(tmp_path: Path):
    csv_path = tmp_path / "telemetry.csv"
    csv_path.write_text(
        "timestamp,latitude,longitude,altitude\n"
        "0.0,28.5567,77.1025,50\n"
        "1.0,28.5568,77.1026,51\n"
        "2.0,28.5569,77.1027,52\n"
    )
    records = parse_telemetry(str(csv_path))
    assert records is not None
    assert len(records) == 3
    assert records[0]["latitude"] == 28.5567


def test_parse_json_telemetry(tmp_path: Path):
    json_path = tmp_path / "telemetry.json"
    data = [
        {"timestamp_s": 0, "latitude": 28.55, "longitude": 77.10, "altitude_m": 50},
        {"timestamp_s": 1, "latitude": 28.56, "longitude": 77.11, "altitude_m": 51},
    ]
    json_path.write_text(json.dumps(data))
    records = parse_telemetry(str(json_path))
    assert records is not None
    assert len(records) == 2


def test_parse_srt_telemetry(tmp_path: Path):
    srt_path = tmp_path / "subtitles.srt"
    srt_path.write_text(
        "1\n"
        "00:00:00,000 --> 00:00:01,000\n"
        "F/2.8, SS 100, ISO 100, GPS (28.5567, 77.1025, 50)\n\n"
        "2\n"
        "00:00:01,000 --> 00:00:02,000\n"
        "F/2.8, SS 100, ISO 100, GPS (28.5568, 77.1026, 51)\n\n"
    )
    records = parse_telemetry(str(srt_path))
    assert records is not None
    assert len(records) == 2
    assert records[0]["latitude"] == 28.5567


def test_telemetry_summary():
    records = [
        {"latitude": 28.55, "longitude": 77.10, "altitude_m": 50},
        {"latitude": 28.56, "longitude": 77.11, "altitude_m": 55},
    ]
    summary = telemetry_summary(records)
    assert summary["available"] is True
    assert summary["has_gps"] is True
    assert summary["count"] == 2
    assert "bbox" in summary

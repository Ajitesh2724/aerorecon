"""Tests for SfM service (COLMAP wrapper & OpenCV fallback)."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.sfm import (
    get_colmap_path,
    _quat_to_rot,
    _save_ply,
    _parse_colmap_stats,
    _parse_colmap_poses,
    run_opencv_sfm,
)


# ── Quaternion & Math Tests ──────────────────────────────────────────


def test_quat_to_rot_identity():
    # w=1, x=0, y=0, z=0
    R = _quat_to_rot(1.0, 0.0, 0.0, 0.0)
    assert np.allclose(R, np.eye(3), atol=1e-6)


def test_quat_to_rot_90_deg_z():
    # 90 degrees around Z axis: w=cos(45)=0.7071068, z=sin(45)=0.7071068
    w = np.cos(np.pi / 4)
    z = np.sin(np.pi / 4)
    R = _quat_to_rot(w, 0.0, 0.0, z)
    expected = np.array([
        [0.0, -1.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ])
    assert np.allclose(R, expected, atol=1e-5)


# ── PLY Export Tests ─────────────────────────────────────────────────


def test_save_ply(tmp_path: Path):
    points = np.array([
        [1.0, 2.0, 3.0],
        [4.5, 5.5, 6.5],
        [-1.2, -3.4, 0.0],
    ], dtype=np.float32)

    ply_file = tmp_path / "test.ply"
    _save_ply(points, ply_file)

    assert ply_file.exists()
    content = ply_file.read_text()
    assert "element vertex 3" in content
    assert "1.000000 2.000000 3.000000" in content
    assert "4.500000 5.500000 6.500000" in content


# ── COLMAP Parser Tests ──────────────────────────────────────────────


def test_parse_colmap_txt(tmp_path: Path):
    txt_dir = tmp_path / "sparse_txt"
    txt_dir.mkdir()

    # Create dummy images.txt
    images_txt = txt_dir / "images.txt"
    images_txt.write_text(
        "# Image list with two lines per image\n"
        "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
        "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
        "1 1.0 0.0 0.0 0.0 0.0 0.0 0.0 1 image_001.jpg\n"
        "100.5 200.5 1 150.2 250.3 2\n"
        "2 1.0 0.0 0.0 0.0 1.0 0.0 0.0 1 image_002.jpg\n"
        "102.1 201.2 1 151.0 252.0 2\n"
    )

    # Create dummy points3D.txt
    points_txt = txt_dir / "points3D.txt"
    points_txt.write_text(
        "# 3D point list with one line per point\n"
        "#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[] as (IMAGE_ID, POINT2D_IDX)\n"
        "1 0.5 1.2 3.4 255 0 0 0.42 1 0 2 0\n"
        "2 1.5 2.2 4.4 0 255 0 0.84 1 1 2 1\n"
    )

    ply_dummy = tmp_path / "sparse.ply"
    ply_dummy.write_text("dummy")

    stats = _parse_colmap_stats(txt_dir, ply_dummy)
    assert stats["registered_images"] == 2
    assert stats["sparse_points"] == 2
    assert abs(stats["mean_reprojection_error"] - 0.63) < 0.01

    poses = _parse_colmap_poses(txt_dir)
    assert len(poses) == 2
    assert poses[0]["image_name"] == "image_001.jpg"
    assert poses[0]["position"] == [0.0, 0.0, 0.0]
    assert poses[1]["image_name"] == "image_002.jpg"


# ── OpenCV SfM Fallback Tests ─────────────────────────────────────────


@pytest.fixture
def feature_rich_image_pair(tmp_path: Path) -> tuple[str, str]:
    """Generate two synthetic frames with textured patterns and slight shift."""
    img_dir = tmp_path / "images"
    img_dir.mkdir()

    h, w = 480, 640
    # Create textured background with many distinct feature points
    np.random.seed(42)
    base = np.zeros((h, w, 3), dtype=np.uint8)

    # Draw grid of textured shapes
    for r in range(40, h - 40, 50):
        for c in range(40, w - 40, 50):
            color = (int(np.random.randint(50, 255)), int(np.random.randint(50, 255)), int(np.random.randint(50, 255)))
            cv2.rectangle(base, (c, r), (c + 30, r + 30), color, -1)
            cv2.circle(base, (c + 15, r + 15), 8, (255, 255, 255), 2)
            cv2.putText(base, f"{r},{c}", (c - 5, r + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    # Frame 1
    p1 = img_dir / "frame_000001.jpg"
    cv2.imwrite(str(p1), base)

    # Frame 2: simulated slight camera movement (affine warp / small translation)
    M = np.float32([[1, 0, -15], [0, 1, -5]])
    frame2 = cv2.warpAffine(base, M, (w, h))
    p2 = img_dir / "frame_000002.jpg"
    cv2.imwrite(str(p2), frame2)

    # Frame 3: simulated further movement
    M2 = np.float32([[1, 0, -30], [0, 1, -10]])
    frame3 = cv2.warpAffine(base, M2, (w, h))
    p3 = img_dir / "frame_000003.jpg"
    cv2.imwrite(str(p3), frame3)

    return str(img_dir)


def test_opencv_sfm(feature_rich_image_pair: str, tmp_path: Path):
    ws_dir = str(tmp_path / "sfm_ws")

    progress_log = []
    def on_progress(pct, msg):
        progress_log.append((pct, msg))

    res = run_opencv_sfm(
        image_dir=feature_rich_image_pair,
        workspace_dir=ws_dir,
        on_progress=on_progress,
    )

    assert res["success"] is True
    assert Path(res["sparse_ply"]).exists()
    assert Path(res["poses_json"]).exists()
    assert res["n_poses"] >= 2
    assert res["stats"]["sparse_points"] >= 0
    assert len(progress_log) > 0

    # Verify poses JSON structure
    poses_data = json.loads(Path(res["poses_json"]).read_text())
    assert len(poses_data) >= 2
    assert "position" in poses_data[0]
    assert "rotation_matrix" in poses_data[0]

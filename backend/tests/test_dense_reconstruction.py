"""Tests for dense reconstruction, 3D Gaussian Splatting, and surface meshing."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.dense_reconstruction import run_dense_reconstruction
from app.services.gaussian_splatting import export_gaussian_splats
from app.services.mesh import generate_surface_mesh


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def synthetic_reconstruction_workspace(tmp_path: Path):
    """Setup synthetic poses, depth maps, and images for testing multi-view fusion."""
    ws = tmp_path / "recon_ws"
    ws.mkdir()

    img_dir = ws / "images"
    depth_dir = ws / "depth"
    img_dir.mkdir()
    depth_dir.mkdir()

    h, w = 120, 160
    poses = []

    for i in range(3):
        stem = f"frame_{i:04d}"

        # Synthetic image
        img = np.zeros((h, w, 3), dtype=np.uint8)
        img[:] = (100 + i * 20, 150, 80)
        img_p = img_dir / f"{stem}.jpg"
        cv2.imwrite(str(img_p), img)

        # Synthetic depth map (planar ground at ~15m)
        depth = np.full((h, w), 15.0 + i * 0.5, dtype=np.float32)
        depth_p = depth_dir / f"depth_{stem}.npy"
        np.save(str(depth_p), depth)

        poses.append({
            "image_name": f"{stem}.jpg",
            "position": [float(i * 2.0), 0.0, 10.0],
            "rotation_matrix": np.eye(3).tolist(),
        })

    poses_json = ws / "camera_poses.json"
    poses_json.write_text(json.dumps(poses, indent=2))

    return {
        "poses_json": str(poses_json),
        "depth_dir": str(depth_dir),
        "image_dir": str(img_dir),
        "ws": ws,
    }


# ── Dense Reconstruction Tests ───────────────────────────────────────


def test_dense_reconstruction(synthetic_reconstruction_workspace: dict, tmp_path: Path):
    out_ply = tmp_path / "dense.ply"

    res = run_dense_reconstruction(
        poses_json_path=synthetic_reconstruction_workspace["poses_json"],
        depth_dir=synthetic_reconstruction_workspace["depth_dir"],
        image_dir=synthetic_reconstruction_workspace["image_dir"],
        output_ply_path=str(out_ply),
        stride=4,
    )

    assert res["success"] is True
    assert out_ply.exists()
    assert res["stats"]["dense_points_count"] > 0

    content = out_ply.read_text()
    assert "format ascii 1.0" in content
    assert "property uchar red" in content
    assert f"element vertex {res['stats']['dense_points_count']}" in content


# ── 3D Gaussian Splatting Tests ──────────────────────────────────────


def test_gaussian_splatting_export(tmp_path: Path):
    out_splat = tmp_path / "gaussian_splat.ply"

    # Create dummy 3D cloud
    np.random.seed(42)
    pts = np.random.uniform(-5, 5, (500, 3)).astype(np.float32)
    cols = np.random.randint(0, 255, (500, 3), dtype=np.uint8)

    res = export_gaussian_splats(
        points=pts,
        colors=cols,
        output_ply_path=str(out_splat),
    )

    assert res["success"] is True
    assert out_splat.exists()
    assert res["stats"]["num_gaussians"] == 500
    assert out_splat.stat().st_size > 1000

    # Verify binary 3DGS header
    with open(out_splat, "rb") as f:
        header = f.read(400).decode("ascii", errors="ignore")
        assert "format binary_little_endian 1.0" in header
        assert "element vertex 500" in header
        assert "property float f_dc_0" in header
        assert "property float opacity" in header
        assert "property float scale_0" in header
        assert "property float rot_0" in header


# ── Surface Mesh Tests ───────────────────────────────────────────────


def test_surface_mesh_generation(tmp_path: Path):
    out_obj = tmp_path / "model.obj"

    # Create dummy 3D terrain grid points
    xs, ys = np.meshgrid(np.linspace(-10, 10, 20), np.linspace(-10, 10, 20))
    zs = np.sin(xs / 3.0) + np.cos(ys / 3.0)

    pts = np.vstack([xs.ravel(), ys.ravel(), zs.ravel()]).T.astype(np.float32)
    cols = np.full_like(pts, 180, dtype=np.uint8)

    res = generate_surface_mesh(
        points=pts,
        colors=cols,
        output_obj_path=str(out_obj),
    )

    assert res["success"] is True
    assert out_obj.exists()
    assert res["stats"]["vertex_count"] == 400
    assert res["stats"]["face_count"] > 0

    content = out_obj.read_text()
    assert "v " in content
    assert "f " in content

"""Tests for learned priors: dynamic object masking, depth estimation, and optical flow."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.services.masking import generate_dynamic_masks
from app.services.depth import estimate_depth_maps
from app.services.optical_flow import compute_sequence_optical_flow


# ── Fixtures ─────────────────────────────────────────────────────────


@pytest.fixture
def test_image_sequence(tmp_path: Path) -> list[str]:
    """Create a sequence of 3 synthetic drone images with texture and motion."""
    seq_dir = tmp_path / "seq"
    seq_dir.mkdir()

    h, w = 240, 320
    np.random.seed(42)

    # Textured ground
    ground = np.random.randint(40, 180, (h + 60, w + 60, 3), dtype=np.uint8)

    paths = []
    for i in range(3):
        # Global camera translation
        dx = i * 6
        dy = i * 2
        frame = ground[dy : dy + h, dx : dx + w].copy()

        # Add a stationary landmark (building)
        cv2.rectangle(frame, (100 - dx, 80 - dy), (140 - dx, 120 - dy), (220, 220, 220), -1)

        # Add an independently moving dynamic object (vehicle moving in opposite direction)
        car_x = 50 + i * 25
        car_y = 150
        cv2.rectangle(frame, (car_x, car_y), (car_x + 30, car_y + 15), (0, 0, 255), -1)

        p = seq_dir / f"frame_{i:04d}.jpg"
        cv2.imwrite(str(p), frame)
        paths.append(str(p))

    return paths


# ── Dynamic Masking Tests ────────────────────────────────────────────


def test_dynamic_masking(test_image_sequence: list[str], tmp_path: Path):
    out_dir = str(tmp_path / "mask_out")

    progress = []
    def on_prog(pct, msg):
        progress.append((pct, msg))

    res = generate_dynamic_masks(
        test_image_sequence,
        output_dir=out_dir,
        on_progress=on_prog,
    )

    assert res["success"] is True
    assert Path(res["masks_dir"]).exists()
    assert Path(res["vis_dir"]).exists()
    assert Path(res["manifest_path"]).exists()

    manifest = json.loads(Path(res["manifest_path"]).read_text())
    assert len(manifest) == len(test_image_sequence)

    # Verify mask properties
    first_mask_path = manifest[0]["mask_path"]
    assert Path(first_mask_path).exists()
    mask_img = cv2.imread(first_mask_path, cv2.IMREAD_GRAYSCALE)
    assert mask_img is not None
    assert mask_img.shape == (240, 320)
    # Mask should contain valid values (0 and/or 255)
    unique_vals = set(np.unique(mask_img))
    assert unique_vals.issubset({0, 255})

    # Verify overlay visualization
    vis_path = manifest[0]["vis_path"]
    assert Path(vis_path).exists()
    vis_img = cv2.imread(vis_path)
    assert vis_img is not None
    assert vis_img.shape == (240, 320, 3)


# ── Depth Estimation Tests ───────────────────────────────────────────


def test_depth_estimation(test_image_sequence: list[str], tmp_path: Path):
    out_dir = str(tmp_path / "depth_out")

    progress = []
    def on_prog(pct, msg):
        progress.append((pct, msg))

    res = estimate_depth_maps(
        test_image_sequence,
        output_dir=out_dir,
        on_progress=on_prog,
    )

    assert res["success"] is True
    assert Path(res["raw_dir"]).exists()
    assert Path(res["vis_dir"]).exists()
    assert Path(res["manifest_path"]).exists()

    manifest = json.loads(Path(res["manifest_path"]).read_text())
    assert len(manifest) == len(test_image_sequence)

    # Check raw depth array
    raw_path = manifest[0]["raw_path"]
    assert Path(raw_path).exists()
    depth_arr = np.load(raw_path)
    assert depth_arr.shape == (240, 320)
    assert depth_arr.dtype == np.float32
    assert np.all(np.isfinite(depth_arr))
    assert np.all(depth_arr > 0)

    # Check colorized visualization
    vis_path = manifest[0]["vis_path"]
    assert Path(vis_path).exists()
    vis_img = cv2.imread(vis_path)
    assert vis_img is not None
    assert vis_img.shape == (240, 320, 3)

    assert res["stats"]["total_depth_maps"] == 3
    assert res["stats"]["mean_scene_depth"] > 0


# ── Optical Flow Tests ───────────────────────────────────────────────


def test_optical_flow(test_image_sequence: list[str], tmp_path: Path):
    out_dir = str(tmp_path / "flow_out")

    progress = []
    def on_prog(pct, msg):
        progress.append((pct, msg))

    res = compute_sequence_optical_flow(
        test_image_sequence,
        output_dir=out_dir,
        on_progress=on_prog,
    )

    assert res["success"] is True
    assert Path(res["raw_dir"]).exists()
    assert Path(res["vis_dir"]).exists()
    assert Path(res["manifest_path"]).exists()

    manifest = json.loads(Path(res["manifest_path"]).read_text())
    assert len(manifest) == 2  # 3 frames -> 2 pairs

    # Check raw flow field
    raw_path = manifest[0]["raw_path"]
    assert Path(raw_path).exists()
    flow_arr = np.load(raw_path)
    assert flow_arr.shape == (240, 320, 2)
    assert flow_arr.dtype == np.float32

    # Check flow visualization
    vis_path = manifest[0]["vis_path"]
    assert Path(vis_path).exists()
    vis_img = cv2.imread(vis_path)
    assert vis_img is not None
    assert vis_img.shape == (240, 320, 3)

    # Check consistency rate & magnitude
    assert 0.0 <= manifest[0]["forward_backward_consistency"] <= 1.0
    assert manifest[0]["mean_magnitude_px"] > 0

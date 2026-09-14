"""3D Gaussian Splatting module — initialization, optimization representation, and export."""

from __future__ import annotations

import logging
import struct
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np

logger = logging.getLogger("aerorecon.splat")

# Spherical harmonics constant for DC term
SH_C0 = 0.28209479177387814


def export_gaussian_splats(
    points: np.ndarray,
    colors: np.ndarray,
    output_ply_path: str,
    max_gaussians: int = 150_000,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Initialize 3D Gaussians from 3D points and colors, and export standard Gaussian PLY.

    Compatible with standard 3DGS WebGL viewers (Three.js 3DGS, Potree, Antimatter15).
    """
    out_ply = Path(output_ply_path)
    out_ply.parent.mkdir(parents=True, exist_ok=True)

    n_raw = len(points)
    if n_raw == 0:
        logger.warning("No points supplied for Gaussian Splatting.")
        # Create a single dummy splat at origin
        points = np.zeros((1, 3), dtype=np.float32)
        colors = np.full((1, 3), 180, dtype=np.uint8)

    # Subsample if exceeds max_gaussians to respect VRAM limits (GTX 1650 4 GB)
    if len(points) > max_gaussians:
        logger.info("Subsampling %d points down to %d Gaussians for 4 GB VRAM cap", len(points), max_gaussians)
        indices = np.random.choice(len(points), max_gaussians, replace=False)
        pts = points[indices].astype(np.float32)
        cols = colors[indices].astype(np.float32)
    else:
        pts = points.astype(np.float32)
        cols = colors.astype(np.float32)

    n = len(pts)
    if on_progress:
        on_progress(20, f"Computing spatial covariances for {n:,} Gaussians…")

    # Estimate initial isotropic scale using nearest neighbor heuristic
    scales = _estimate_initial_scales(pts)

    if on_progress:
        on_progress(50, "Formatting spherical harmonics and orientations…")

    # Rotations: identity quaternion [qw=1, qx=0, qy=0, qz=0]
    rotations = np.zeros((n, 4), dtype=np.float32)
    rotations[:, 0] = 1.0

    # Opacity: initialize to ~0.8 (in logit space: logit(0.8) ~ 1.386)
    opacities = np.full((n, 1), 1.386, dtype=np.float32)

    # Convert RGB [0..255] to spherical harmonics DC coefficients: (RGB/255 - 0.5) / SH_C0
    sh_dc = (cols / 255.0 - 0.5) / SH_C0

    # Log scales: log(scale)
    log_scales = np.log(np.maximum(scales, 1e-4))
    log_scales_3d = np.repeat(log_scales[:, None], 3, axis=1)

    if on_progress:
        on_progress(80, "Writing standard 3D Gaussian PLY file…")

    # Write binary 3DGS PLY
    _write_gaussian_ply(
        filepath=out_ply,
        positions=pts,
        sh_dc=sh_dc,
        opacities=opacities,
        scales=log_scales_3d,
        rotations=rotations,
    )

    if on_progress:
        on_progress(100, "Gaussian Splatting export complete.")

    stats = {
        "num_gaussians": n,
        "ply_size_mb": round(out_ply.stat().st_size / (1024 * 1024), 2),
        "mean_scale_m": round(float(np.mean(scales)), 4),
    }

    logger.info("Exported %d 3D Gaussians to %s (%.1f MB)", n, out_ply, stats["ply_size_mb"])
    return {
        "success": True,
        "splat_ply": str(out_ply),
        "stats": stats,
    }


def _estimate_initial_scales(pts: np.ndarray) -> np.ndarray:
    """Estimate initial Gaussian radius based on local point density."""
    n = len(pts)
    if n <= 1:
        return np.full(n, 0.05, dtype=np.float32)

    # Sample a subset to estimate average density quickly
    sample_size = min(n, 2000)
    sample_indices = np.random.choice(n, sample_size, replace=False)
    sample_pts = pts[sample_indices]

    # Compute bounding box diagonal
    bbox_min = pts.min(axis=0)
    bbox_max = pts.max(axis=0)
    diag = float(np.linalg.norm(bbox_max - bbox_min))

    # Base scale heuristic: average spacing in volume
    vol = max(1e-3, (bbox_max[0] - bbox_min[0]) * (bbox_max[1] - bbox_min[1]) * max(0.1, bbox_max[2] - bbox_min[2]))
    avg_spacing = (vol / max(1, n)) ** (1.0 / 3.0)

    # Clamp scale to sensible limits (e.g. 1cm to 50cm)
    base_scale = float(np.clip(avg_spacing * 1.2, 0.01, 0.5))
    return np.full(n, base_scale, dtype=np.float32)


def _write_gaussian_ply(
    filepath: Path,
    positions: np.ndarray,
    sh_dc: np.ndarray,
    opacities: np.ndarray,
    scales: np.ndarray,
    rotations: np.ndarray,
) -> None:
    """Write binary little-endian PLY matching the official 3D Gaussian Splatting schema."""
    n = len(positions)
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property float nx\n"
        "property float ny\n"
        "property float nz\n"
        "property float f_dc_0\n"
        "property float f_dc_1\n"
        "property float f_dc_2\n"
        "property float opacity\n"
        "property float scale_0\n"
        "property float scale_1\n"
        "property float scale_2\n"
        "property float rot_0\n"
        "property float rot_1\n"
        "property float rot_2\n"
        "property float rot_3\n"
        "end_header\n"
    )

    normals = np.zeros((n, 3), dtype=np.float32)

    # Combine into single structured array for fast binary write
    # 3 (xyz) + 3 (normals) + 3 (sh_dc) + 1 (opacity) + 3 (scales) + 4 (rot) = 17 floats per vertex
    dtype = [
        ("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
        ("nx", "<f4"), ("ny", "<f4"), ("nz", "<f4"),
        ("f_dc_0", "<f4"), ("f_dc_1", "<f4"), ("f_dc_2", "<f4"),
        ("opacity", "<f4"),
        ("scale_0", "<f4"), ("scale_1", "<f4"), ("scale_2", "<f4"),
        ("rot_0", "<f4"), ("rot_1", "<f4"), ("rot_2", "<f4"), ("rot_3", "<f4"),
    ]

    elements = np.empty(n, dtype=dtype)
    elements["x"] = positions[:, 0]
    elements["y"] = positions[:, 1]
    elements["z"] = positions[:, 2]
    elements["nx"] = normals[:, 0]
    elements["ny"] = normals[:, 1]
    elements["nz"] = normals[:, 2]
    elements["f_dc_0"] = sh_dc[:, 0]
    elements["f_dc_1"] = sh_dc[:, 1]
    elements["f_dc_2"] = sh_dc[:, 2]
    elements["opacity"] = opacities[:, 0]
    elements["scale_0"] = scales[:, 0]
    elements["scale_1"] = scales[:, 1]
    elements["scale_2"] = scales[:, 2]
    elements["rot_0"] = rotations[:, 0]
    elements["rot_1"] = rotations[:, 1]
    elements["rot_2"] = rotations[:, 2]
    elements["rot_3"] = rotations[:, 3]

    with open(filepath, "wb") as f:
        f.write(header.encode("ascii"))
        elements.tofile(f)

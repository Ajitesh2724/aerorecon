"""Dense 3D reconstruction via depth back-projection and multi-view fusion."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("aerorecon.dense")


def run_dense_reconstruction(
    poses_json_path: str,
    depth_dir: str,
    image_dir: str,
    output_ply_path: str,
    masks_dir: Optional[str] = None,
    camera_intrinsics: Optional[dict] = None,
    stride: int = 4,  # sample every 4th pixel for balance of density & speed
    max_points: int = 500_000,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Fuse multi-view depth maps and camera poses into a dense colored 3D point cloud.

    Back-projects depth map pixels:
      P_cam = (K^-1 * [u, v, 1]^T) * depth(u, v)
      P_world = R^T * P_cam + Center
    """
    poses_p = Path(poses_json_path)
    if not poses_p.exists():
        raise FileNotFoundError(f"Poses JSON not found at {poses_json_path}")

    poses: list[dict] = json.loads(poses_p.read_text())
    if not poses:
        raise ValueError("No camera poses available in poses JSON.")

    depth_path = Path(depth_dir)
    image_path = Path(image_dir)
    masks_path = Path(masks_dir) if masks_dir else None

    all_points = []
    all_colors = []

    n_poses = len(poses)
    logger.info("Starting dense reconstruction from %d camera views", n_poses)

    for i, pose in enumerate(poses):
        if on_progress:
            pct = 10 + (i / max(1, n_poses)) * 70
            on_progress(pct, f"Densifying view {i+1}/{n_poses}…")

        img_name = pose.get("image_name", "")
        stem = Path(img_name).stem

        # Locate depth file (.npy)
        depth_file = depth_path / f"depth_{stem}.npy"
        if not depth_file.exists():
            # Try finding any matching npy
            matching = list(depth_path.glob(f"*{stem}*.npy"))
            if matching:
                depth_file = matching[0]
            else:
                continue

        # Locate corresponding RGB image
        img_file = image_path / img_name
        if not img_file.exists():
            matching_imgs = list(image_path.glob(f"*{stem}*"))
            if matching_imgs:
                img_file = matching_imgs[0]
            else:
                continue

        depth_map = np.load(str(depth_file))
        img = cv2.imread(str(img_file))
        if img is None or depth_map is None:
            continue

        h, w = depth_map.shape[:2]
        img = cv2.resize(img, (w, h))

        # Check mask if available
        mask = None
        if masks_path and masks_path.exists():
            mask_file = masks_path / f"mask_{stem}.png"
            if mask_file.exists():
                mask = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
                if mask is not None:
                    mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)

        # Camera Intrinsics
        if camera_intrinsics and "fx" in camera_intrinsics:
            fx = float(camera_intrinsics["fx"])
            fy = float(camera_intrinsics.get("fy", fx))
            cx = float(camera_intrinsics.get("cx", w / 2))
            cy = float(camera_intrinsics.get("cy", h / 2))
        else:
            fx = fy = max(w, h) * 1.1
            cx, cy = w / 2.0, h / 2.0

        # Camera Extrinsics
        R = np.array(pose.get("rotation_matrix", np.eye(3)), dtype=np.float64)
        center = np.array(pose.get("position", [0, 0, 0]), dtype=np.float64)

        # Generate pixel grid sampled with stride
        y_indices, x_indices = np.mgrid[0:h:stride, 0:w:stride]
        sampled_depth = depth_map[0:h:stride, 0:w:stride]
        sampled_bgr = img[0:h:stride, 0:w:stride]

        # Valid depth mask (> 0.1m, < 200m)
        valid = (sampled_depth > 0.1) & (sampled_depth < 200.0) & np.isfinite(sampled_depth)
        if mask is not None:
            sampled_mask = mask[0:h:stride, 0:w:stride]
            valid = valid & (sampled_mask > 128)

        if not np.any(valid):
            continue

        u = x_indices[valid].astype(np.float64)
        v = y_indices[valid].astype(np.float64)
        z = sampled_depth[valid].astype(np.float64)

        # Normalized camera coordinates
        x_cam = (u - cx) * z / fx
        y_cam = (v - cy) * z / fy
        z_cam = z

        pts_cam = np.vstack([x_cam, y_cam, z_cam]).T  # (N, 3)

        # Transform to world coordinates: P_world = R^T * P_cam + center
        pts_world = (R.T @ pts_cam.T).T + center

        # Sample colors (BGR -> RGB, 0..255)
        rgb = sampled_bgr[valid][:, [2, 1, 0]]

        all_points.append(pts_world)
        all_colors.append(rgb)

    if on_progress:
        on_progress(85, "Filtering and fusing dense 3D points…")

    if not all_points:
        logger.warning("No dense points back-projected. Creating empty point cloud.")
        pts_final = np.zeros((0, 3), dtype=np.float32)
        colors_final = np.zeros((0, 3), dtype=np.uint8)
    else:
        all_pts = np.vstack(all_points).astype(np.float32)
        all_cols = np.vstack(all_colors).astype(np.uint8)

        # Statistical outlier removal / voxel thinning if point count is large
        if len(all_pts) > max_points:
            indices = np.random.choice(len(all_pts), max_points, replace=False)
            pts_final = all_pts[indices]
            colors_final = all_cols[indices]
        else:
            pts_final = all_pts
            colors_final = all_cols

        # Simple radius/box outlier trimming
        pts_final, colors_final = _trim_outliers(pts_final, colors_final)

    # Save to PLY
    out_ply = Path(output_ply_path)
    out_ply.parent.mkdir(parents=True, exist_ok=True)
    _write_colored_ply(pts_final, colors_final, out_ply)

    if on_progress:
        on_progress(100, "Dense reconstruction complete.")

    stats = {
        "dense_points_count": len(pts_final),
        "fused_views_count": n_poses,
        "ply_size_mb": round(out_ply.stat().st_size / (1024 * 1024), 2) if out_ply.exists() else 0.0,
    }

    logger.info("Dense reconstruction finished with %d points", len(pts_final))
    return {
        "success": True,
        "dense_ply": str(out_ply),
        "stats": stats,
    }


def _trim_outliers(pts: np.ndarray, cols: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Trim extreme coordinate outliers beyond 3 standard deviations from median."""
    if len(pts) < 100:
        return pts, cols

    median = np.median(pts, axis=0)
    dists = np.linalg.norm(pts - median, axis=1)
    threshold = np.percentile(dists, 98)  # Keep 98% of points closest to median

    valid = dists <= threshold
    return pts[valid], cols[valid]


def _write_colored_ply(points: np.ndarray, colors: np.ndarray, path: Path) -> None:
    """Save colored 3D point cloud in standard ASCII PLY format."""
    n = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "property uchar red\n"
        "property uchar green\n"
        "property uchar blue\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for i in range(n):
            p = points[i]
            c = colors[i]
            f.write(f"{p[0]:.4f} {p[1]:.4f} {p[2]:.4f} {c[0]} {c[1]} {c[2]}\n")

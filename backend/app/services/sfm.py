"""COLMAP Structure-from-Motion integration with OpenCV fallback."""

from __future__ import annotations

import json
import logging
import shutil
import subprocess
import struct
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger("aerorecon.sfm")


# ═══════════════════════════════════════════════════════════════════
#  COLMAP runner
# ═══════════════════════════════════════════════════════════════════


def get_colmap_path() -> Optional[str]:
    """Return the COLMAP binary path or None if unavailable."""
    custom = settings.COLMAP_PATH
    if custom and Path(custom).exists():
        return custom
    found = shutil.which("colmap")
    return found


def run_colmap_sfm(
    image_dir: str,
    workspace_dir: str,
    camera_intrinsics: Optional[dict] = None,
    on_progress: Optional[callable] = None,
) -> dict[str, Any]:
    """Run the full COLMAP SfM pipeline and return reconstruction stats.

    Steps: feature_extractor → exhaustive_matcher → mapper → model_converter
    Returns dict with poses, sparse cloud path, and statistics.
    """
    colmap = get_colmap_path()
    if not colmap:
        raise RuntimeError("COLMAP is not installed or not found in PATH.")

    ws = Path(workspace_dir)
    db_path = ws / "database.db"
    sparse_dir = ws / "sparse"
    sparse_dir.mkdir(parents=True, exist_ok=True)

    # Count images
    image_path = Path(image_dir)
    image_files = sorted(
        [f for f in image_path.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    n_images = len(image_files)
    if n_images < 3:
        raise ValueError(f"Need at least 3 images for SfM, got {n_images}.")

    logger.info("Starting COLMAP SfM on %d images", n_images)

    # ── Feature extraction ────────────────────────────────────
    if on_progress:
        on_progress(10, "Extracting features…")

    fe_cmd = [
        colmap, "feature_extractor",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--ImageReader.single_camera", "1",
        "--SiftExtraction.max_image_size", "1600",
        "--SiftExtraction.max_num_features", "8192",
    ]
    if camera_intrinsics and "fx" in camera_intrinsics:
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics.get("fy", fx)
        cx = camera_intrinsics.get("cx", 0)
        cy = camera_intrinsics.get("cy", 0)
        fe_cmd += [
            "--ImageReader.camera_model", "PINHOLE",
            "--ImageReader.camera_params", f"{fx},{fy},{cx},{cy}",
        ]

    _run_subprocess(fe_cmd, "feature_extractor")

    # ── Feature matching ──────────────────────────────────────
    if on_progress:
        on_progress(30, "Matching features…")

    match_type = "sequential_matcher" if n_images > 100 else "exhaustive_matcher"
    match_cmd = [
        colmap, match_type,
        "--database_path", str(db_path),
    ]
    if match_type == "sequential_matcher":
        match_cmd += ["--SequentialMatching.overlap", "10"]

    _run_subprocess(match_cmd, match_type)

    # ── Incremental mapper ────────────────────────────────────
    if on_progress:
        on_progress(50, "Reconstructing scene…")

    mapper_cmd = [
        colmap, "mapper",
        "--database_path", str(db_path),
        "--image_path", str(image_dir),
        "--output_path", str(sparse_dir),
        "--Mapper.ba_refine_focal_length", "1",
        "--Mapper.ba_refine_extra_params", "1",
    ]
    _run_subprocess(mapper_cmd, "mapper")

    # Find the best reconstruction (COLMAP may produce multiple)
    recon_dirs = sorted(sparse_dir.iterdir())
    if not recon_dirs:
        raise RuntimeError("COLMAP mapper produced no reconstruction.")
    best_recon = recon_dirs[0]  # 0 is typically the largest

    # ── Export to text format ─────────────────────────────────
    if on_progress:
        on_progress(70, "Exporting model…")

    txt_dir = ws / "sparse_txt"
    txt_dir.mkdir(exist_ok=True)
    export_cmd = [
        colmap, "model_converter",
        "--input_path", str(best_recon),
        "--output_path", str(txt_dir),
        "--output_type", "TXT",
    ]
    _run_subprocess(export_cmd, "model_converter")

    # ── Export to PLY ─────────────────────────────────────────
    if on_progress:
        on_progress(80, "Exporting point cloud…")

    ply_path = ws / "sparse.ply"
    ply_cmd = [
        colmap, "model_converter",
        "--input_path", str(best_recon),
        "--output_path", str(ply_path),
        "--output_type", "PLY",
    ]
    _run_subprocess(ply_cmd, "model_converter_ply")

    # ── Parse results ─────────────────────────────────────────
    if on_progress:
        on_progress(90, "Parsing reconstruction…")

    stats = _parse_colmap_stats(txt_dir, ply_path)
    poses = _parse_colmap_poses(txt_dir)

    # Save poses as JSON for the viewer
    poses_json_path = ws / "camera_poses.json"
    poses_json_path.write_text(json.dumps(poses, indent=2))

    if on_progress:
        on_progress(100, "SfM complete.")

    return {
        "success": True,
        "sparse_ply": str(ply_path),
        "poses_json": str(poses_json_path),
        "sparse_dir": str(best_recon),
        "txt_dir": str(txt_dir),
        "stats": stats,
        "n_poses": len(poses),
    }


def _run_subprocess(cmd: list[str], stage: str) -> None:
    """Execute a COLMAP command and log output."""
    logger.debug("Running: %s", " ".join(cmd))
    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=1800,  # 30 min timeout
    )
    if result.returncode != 0:
        logger.error("COLMAP %s failed:\n%s", stage, result.stderr[:2000])
        raise RuntimeError(f"COLMAP {stage} failed: {result.stderr[:500]}")
    logger.debug("COLMAP %s completed.", stage)


def _parse_colmap_stats(txt_dir: Path, ply_path: Path) -> dict[str, Any]:
    """Extract statistics from COLMAP text output."""
    stats: dict[str, Any] = {}

    # Count cameras, images, points from text files
    images_file = txt_dir / "images.txt"
    points_file = txt_dir / "points3D.txt"

    if images_file.exists():
        lines = images_file.read_text().strip().split("\n")
        # Filter comments and count image entries (every other non-comment line)
        data_lines = [l for l in lines if not l.startswith("#")]
        stats["registered_images"] = len(data_lines) // 2  # image + points lines alternate

    if points_file.exists():
        lines = points_file.read_text().strip().split("\n")
        data_lines = [l for l in lines if not l.startswith("#")]
        stats["sparse_points"] = len(data_lines)

        # Parse mean reprojection error
        errors = []
        for line in data_lines:
            parts = line.split()
            if len(parts) >= 8:
                try:
                    errors.append(float(parts[7]))  # error is 8th column
                except (ValueError, IndexError):
                    pass
        if errors:
            stats["mean_reprojection_error"] = round(float(np.mean(errors)), 4)
            stats["median_reprojection_error"] = round(float(np.median(errors)), 4)

    if ply_path.exists():
        stats["ply_size_mb"] = round(ply_path.stat().st_size / (1024 * 1024), 2)

    return stats


def _parse_colmap_poses(txt_dir: Path) -> list[dict[str, Any]]:
    """Parse camera poses from COLMAP's images.txt format."""
    images_file = txt_dir / "images.txt"
    if not images_file.exists():
        return []

    poses = []
    lines = images_file.read_text().strip().split("\n")
    data_lines = [l for l in lines if not l.startswith("#")]

    for i in range(0, len(data_lines), 2):
        if i >= len(data_lines):
            break
        parts = data_lines[i].split()
        if len(parts) < 10:
            continue

        try:
            image_id = int(parts[0])
            qw, qx, qy, qz = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            tx, ty, tz = float(parts[5]), float(parts[6]), float(parts[7])
            camera_id = int(parts[8])
            name = parts[9]

            # Convert quaternion+translation to camera position
            # R = quat_to_rotation, camera_center = -R^T * t
            R = _quat_to_rot(qw, qx, qy, qz)
            center = (-R.T @ np.array([tx, ty, tz])).tolist()

            poses.append({
                "image_id": image_id,
                "image_name": name,
                "camera_id": camera_id,
                "quaternion": [qw, qx, qy, qz],
                "translation": [tx, ty, tz],
                "position": center,
                "rotation_matrix": R.tolist(),
            })
        except (ValueError, IndexError):
            continue

    return poses


def _quat_to_rot(w: float, x: float, y: float, z: float) -> np.ndarray:
    """Convert quaternion to 3×3 rotation matrix."""
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)],
        [2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)],
    ])
    return R


# ═══════════════════════════════════════════════════════════════════
#  OpenCV SfM fallback (when COLMAP is unavailable)
# ═══════════════════════════════════════════════════════════════════


def run_opencv_sfm(
    image_dir: str,
    workspace_dir: str,
    camera_intrinsics: Optional[dict] = None,
    on_progress: Optional[callable] = None,
) -> dict[str, Any]:
    """Basic two-view SfM using OpenCV feature matching and essential matrix.

    This is a simplified fallback — not as robust as COLMAP, but works
    without any external dependencies.
    """
    ws = Path(workspace_dir)
    ws.mkdir(parents=True, exist_ok=True)

    image_path = Path(image_dir)
    image_files = sorted(
        [f for f in image_path.iterdir()
         if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    )
    n_images = len(image_files)
    if n_images < 2:
        raise ValueError(f"Need at least 2 images for SfM, got {n_images}.")

    logger.info("Running OpenCV SfM fallback on %d images", n_images)
    if on_progress:
        on_progress(10, "Detecting features…")

    # Read images and detect features
    sift = cv2.SIFT_create(nfeatures=4096)
    bf = cv2.BFMatcher(cv2.NORM_L2)

    images_data = []
    for f in image_files:
        img = cv2.imread(str(f))
        if img is None:
            continue
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kp, desc = sift.detectAndCompute(gray, None)
        images_data.append({
            "file": f,
            "shape": img.shape[:2],
            "keypoints": kp,
            "descriptors": desc,
        })

    if len(images_data) < 2:
        raise ValueError("Not enough images could be read.")

    # Camera matrix estimate
    h, w = images_data[0]["shape"]
    if camera_intrinsics and "fx" in camera_intrinsics:
        fx = camera_intrinsics["fx"]
        fy = camera_intrinsics.get("fy", fx)
        cx = camera_intrinsics.get("cx", w / 2)
        cy = camera_intrinsics.get("cy", h / 2)
    else:
        # Estimate focal length from image size
        fx = fy = max(w, h) * 1.2
        cx, cy = w / 2, h / 2

    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)

    if on_progress:
        on_progress(30, "Matching features between frames…")

    # Incremental SfM: chain pairwise reconstructions
    all_points_3d = []
    poses = []

    # First camera at origin
    poses.append({
        "image_name": images_data[0]["file"].name,
        "position": [0.0, 0.0, 0.0],
        "rotation_matrix": np.eye(3).tolist(),
    })

    for i in range(len(images_data) - 1):
        if on_progress:
            pct = 30 + (i / max(1, len(images_data) - 2)) * 50
            on_progress(pct, f"Processing pair {i+1}/{len(images_data)-1}…")

        desc1 = images_data[i]["descriptors"]
        desc2 = images_data[i + 1]["descriptors"]
        kp1 = images_data[i]["keypoints"]
        kp2 = images_data[i + 1]["keypoints"]

        if desc1 is None or desc2 is None:
            continue

        # Match features
        matches = bf.knnMatch(desc1, desc2, k=2)
        # Ratio test
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 15:
            logger.warning("Too few matches (%d) between frames %d-%d", len(good), i, i + 1)
            continue

        pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

        # Essential matrix
        E, mask = cv2.findEssentialMat(pts1, pts2, K, method=cv2.RANSAC, threshold=1.0)
        if E is None:
            continue

        _, R, t, mask_pose = cv2.recoverPose(E, pts1, pts2, K)

        # Triangulate points
        P1 = K @ np.hstack([np.eye(3), np.zeros((3, 1))])
        P2 = K @ np.hstack([R, t])
        pts_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
        pts_3d = pts_4d[:3] / pts_4d[3:4]

        # Filter by reprojection and depth
        valid = (pts_3d[2] > 0) & (pts_3d[2] < 100)
        points = pts_3d[:, valid].T

        if len(points) > 0:
            all_points_3d.append(points)

        # Camera position for image i+1
        center = (-R.T @ t).flatten().tolist()
        poses.append({
            "image_name": images_data[i + 1]["file"].name,
            "position": center,
            "rotation_matrix": R.tolist(),
        })

    if on_progress:
        on_progress(85, "Exporting point cloud…")

    # Combine all 3D points
    if all_points_3d:
        all_pts = np.vstack(all_points_3d)
    else:
        all_pts = np.zeros((0, 3))

    # Save as PLY
    ply_path = ws / "sparse.ply"
    _save_ply(all_pts, ply_path)

    # Save poses as JSON
    poses_json_path = ws / "camera_poses.json"
    poses_json_path.write_text(json.dumps(poses, indent=2))

    stats = {
        "registered_images": len(poses),
        "sparse_points": len(all_pts),
        "method": "opencv_fallback",
    }

    if on_progress:
        on_progress(100, "OpenCV SfM complete.")

    logger.info("OpenCV SfM: %d poses, %d points", len(poses), len(all_pts))
    return {
        "success": True,
        "sparse_ply": str(ply_path),
        "poses_json": str(poses_json_path),
        "stats": stats,
        "n_poses": len(poses),
    }


def _save_ply(points: np.ndarray, path: Path) -> None:
    """Save a Nx3 point array as a simple PLY file."""
    n = len(points)
    header = (
        "ply\n"
        "format ascii 1.0\n"
        f"element vertex {n}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        "end_header\n"
    )
    with open(path, "w") as f:
        f.write(header)
        for p in points:
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f}\n")
    logger.debug("Saved PLY with %d points to %s", n, path)

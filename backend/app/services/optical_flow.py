"""Optical flow estimation service for continuous motion priors and consistency checks."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("aerorecon.flow")


def compute_sequence_optical_flow(
    image_paths: list[str],
    output_dir: str,
    method: str = "dis",  # 'dis' (Dense Inverse Search) or 'farneback'
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Compute dense optical flow between consecutive image pairs.

    Computes:
      - Forward flow (t -> t+1)
      - Backward flow (t+1 -> t)
      - Forward-backward consistency mask (reliable vs. occluded/dynamic pixels)

    Saves:
      - .npy raw flow fields (H x W x 2) in `output_dir/raw`
      - .png color-coded flow visualizations in `output_dir/visualizations`

    Returns summary stats and artifact paths.
    """
    out = Path(output_dir)
    raw_dir = out / "raw"
    vis_dir = out / "visualizations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(image_paths)
    if n_images < 2:
        return {
            "success": True,
            "raw_dir": str(raw_dir),
            "vis_dir": str(vis_dir),
            "manifest": [],
            "stats": {"total_pairs": 0},
        }

    n_pairs = n_images - 1
    manifest: list[dict[str, Any]] = []
    all_magnitudes: list[float] = []
    all_consistency_rates: list[float] = []

    # Initialize optical flow engine
    dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)

    for i in range(n_pairs):
        if on_progress:
            pct = 10 + (i / max(1, n_pairs)) * 80
            on_progress(pct, f"Computing optical flow: pair {i+1}/{n_pairs}…")

        p1_str = image_paths[i]
        p2_str = image_paths[i + 1]

        p1 = Path(p1_str)
        p2 = Path(p2_str)

        img1 = cv2.imread(str(p1))
        img2 = cv2.imread(str(p2))

        if img1 is None or img2 is None:
            continue

        g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        h, w = g1.shape

        # 1. Forward flow: g1 -> g2
        flow_fw = dis.calc(g1, g2, None)

        # 2. Backward flow: g2 -> g1
        flow_bw = dis.calc(g2, g1, None)

        # 3. Forward-backward consistency check
        consistency_mask, mean_fb_error = _forward_backward_check(flow_fw, flow_bw)
        consistency_rate = float(np.mean(consistency_mask))
        all_consistency_rates.append(consistency_rate)

        # 4. Magnitude and angle
        mag, ang = cv2.cartToPolar(flow_fw[..., 0], flow_fw[..., 1])
        mean_mag = float(np.mean(mag))
        all_magnitudes.append(mean_mag)

        # 5. Save raw flow .npy
        flow_filename = f"flow_{p1.stem}_to_{p2.stem}.npy"
        raw_path = raw_dir / flow_filename
        np.save(str(raw_path), flow_fw.astype(np.float32))

        # 6. Save visualization (HSV color wheel)
        vis_flow = _flow_to_color(flow_fw)
        vis_filename = f"flow_vis_{p1.stem}_to_{p2.stem}.png"
        vis_path = vis_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_flow)

        manifest.append({
            "frame_a": p1.name,
            "frame_b": p2.name,
            "raw_path": str(raw_path),
            "vis_path": str(vis_path),
            "mean_magnitude_px": round(mean_mag, 3),
            "forward_backward_consistency": round(consistency_rate, 4),
            "mean_fb_error_px": round(mean_fb_error, 3),
        })

    manifest_json = out / "flow_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    if on_progress:
        on_progress(100, "Optical flow computation complete.")

    stats = {
        "total_pairs": len(manifest),
        "mean_flow_velocity_px": round(float(np.mean(all_magnitudes)), 3) if all_magnitudes else 0.0,
        "mean_consistency_rate": round(float(np.mean(all_consistency_rates)), 4) if all_consistency_rates else 0.0,
        "method": "opencv_dis_optical_flow",
    }

    logger.info("Optical flow completed: %d pairs processed", len(manifest))
    return {
        "success": True,
        "raw_dir": str(raw_dir),
        "vis_dir": str(vis_dir),
        "manifest_path": str(manifest_json),
        "stats": stats,
    }


def _forward_backward_check(flow_fw: np.ndarray, flow_bw: np.ndarray, threshold: float = 2.0):
    """Compute forward-backward consistency check.

    Warp backward flow using forward flow, and check if (flow_fw + warped_bw) ~ 0.
    Returns (binary_mask, mean_error).
    """
    h, w = flow_fw.shape[:2]
    # Grid coordinates
    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))

    # Target points reached by forward flow
    target_x = np.clip(x_coords + flow_fw[..., 0], 0, w - 1).astype(np.float32)
    target_y = np.clip(y_coords + flow_fw[..., 1], 0, h - 1).astype(np.float32)

    # Sample backward flow at target points
    sampled_bw_x = cv2.remap(flow_bw[..., 0], target_x, target_y, interpolation=cv2.INTER_LINEAR)
    sampled_bw_y = cv2.remap(flow_bw[..., 1], target_x, target_y, interpolation=cv2.INTER_LINEAR)

    # Difference should be close to zero
    diff_x = flow_fw[..., 0] + sampled_bw_x
    diff_y = flow_fw[..., 1] + sampled_bw_y
    fb_error = np.sqrt(diff_x ** 2 + diff_y ** 2)

    valid_mask = (fb_error <= threshold).astype(np.uint8)
    mean_err = float(np.mean(fb_error))
    return valid_mask, mean_err


def _flow_to_color(flow: np.ndarray) -> np.ndarray:
    """Standard optical flow visualization using the HSV color wheel.

    Direction = Hue, Magnitude = Value, Saturation = 255.
    """
    h, w = flow.shape[:2]
    hsv = np.zeros((h, w, 3), dtype=np.uint8)
    hsv[..., 1] = 255

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    # Angle to 0..180 for OpenCV HSV Hue
    hsv[..., 0] = ang * 180 / np.pi / 2
    # Normalize magnitude to 0..255 Value
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)

    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return rgb

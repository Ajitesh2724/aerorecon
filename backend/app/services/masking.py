"""Dynamic object masking for drone footage.

Filters out transient elements (vehicles, pedestrians, animals) so they do
not corrupt Structure-from-Motion or dense reconstruction.
Supports YOLO-based semantic masking when available, with an automatic
motion-compensated temporal difference fallback.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("aerorecon.masking")


DYNAMIC_CLASSES = frozenset({
    "person", "car", "truck", "bus", "motorcycle", "bicycle", "boat",
    "airplane", "train", "dog", "horse", "sheep", "cow", "cat",
})


def generate_dynamic_masks(
    image_paths: list[str],
    output_dir: str,
    confidence_threshold: float = 0.35,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Generate binary masks for moving/transient objects in a sequence of frames.

    Mask convention:
      - 255 (white): static scene (valid for 3D reconstruction)
      - 0 (black): dynamic/transient object (excluded from matching)

    Returns summary dictionary with paths and statistics.
    """
    out = Path(output_dir)
    masks_dir = out / "masks"
    vis_dir = out / "visualizations"
    masks_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(image_paths)
    if n_images == 0:
        return {
            "success": True,
            "masks_dir": str(masks_dir),
            "vis_dir": str(vis_dir),
            "manifest": [],
            "stats": {"total": 0, "masked_count": 0, "mean_masked_ratio": 0.0},
        }

    # Check for YOLO detector
    detector = _init_detector()

    manifest: list[dict[str, Any]] = []
    masked_count = 0
    total_ratios: list[float] = []

    for i, img_path_str in enumerate(image_paths):
        if on_progress:
            pct = 10 + (i / max(1, n_images)) * 80
            on_progress(pct, f"Masking dynamic objects: frame {i+1}/{n_images}…")

        img_p = Path(img_path_str)
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        h, w = img.shape[:2]
        # Start with all-valid mask (255)
        mask = np.full((h, w), 255, dtype=np.uint8)

        detections = []
        if detector:
            detections = _detect_with_yolo(detector, img, confidence_threshold)
            for box in detections:
                x1, y1, x2, y2 = box["bbox"]
                # Mark bounding box region as dynamic (0)
                mask[y1:y2, x1:x2] = 0
        else:
            # Motion compensation fallback using adjacent frame if available
            prev_img = None
            if i > 0:
                prev_img = cv2.imread(image_paths[i - 1])
            elif i + 1 < n_images:
                prev_img = cv2.imread(image_paths[i + 1])

            if prev_img is not None:
                diff_mask = _detect_motion_differences(prev_img, img)
                mask = cv2.bitwise_and(mask, cv2.bitwise_not(diff_mask))

        # Calculate statistics
        dynamic_pixels = int(np.sum(mask == 0))
        total_pixels = h * w
        masked_ratio = dynamic_pixels / max(1, total_pixels)
        total_ratios.append(masked_ratio)

        has_dynamic = masked_ratio > 0.005
        if has_dynamic:
            masked_count += 1

        # Save binary mask
        mask_filename = f"mask_{img_p.stem}.png"
        mask_path = masks_dir / mask_filename
        cv2.imwrite(str(mask_path), mask)

        # Save visual overlay
        vis_img = img.copy()
        # Tint dynamic regions in semi-transparent red
        dynamic_overlay = vis_img.copy()
        dynamic_overlay[mask == 0] = [0, 0, 220]
        vis_img = cv2.addWeighted(vis_img, 0.75, dynamic_overlay, 0.25, 0)

        # Add border contour
        contours, _ = cv2.findContours(cv2.bitwise_not(mask), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis_img, contours, -1, (0, 0, 255), 2)

        vis_filename = f"vis_{img_p.stem}.jpg"
        vis_path = vis_dir / vis_filename
        cv2.imwrite(str(vis_path), vis_img)

        manifest.append({
            "image_name": img_p.name,
            "mask_path": str(mask_path),
            "vis_path": str(vis_path),
            "masked_ratio": round(masked_ratio, 4),
            "has_dynamic_objects": has_dynamic,
            "detections_count": len(detections),
        })

    manifest_json = out / "masks_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    if on_progress:
        on_progress(100, "Masking complete.")

    mean_ratio = float(np.mean(total_ratios)) if total_ratios else 0.0
    stats = {
        "total_frames": n_images,
        "frames_with_dynamic_objects": masked_count,
        "mean_masked_ratio": round(mean_ratio, 4),
        "method": "yolo" if detector else "motion_compensated_difference",
    }

    logger.info("Dynamic masking completed: %d/%d frames with dynamic objects", masked_count, n_images)
    return {
        "success": True,
        "masks_dir": str(masks_dir),
        "vis_dir": str(vis_dir),
        "manifest_path": str(manifest_json),
        "stats": stats,
    }


def _init_detector():
    """Attempt to import and initialize YOLOv8 if installed."""
    try:
        from ultralytics import YOLO  # type: ignore
        model = YOLO("yolov8n.pt")
        logger.info("Loaded YOLOv8 model for dynamic object masking.")
        return model
    except Exception as exc:
        logger.debug("Ultralytics/YOLO not available (%s); using motion fallback.", exc)
        return None


def _detect_with_yolo(detector, img: np.ndarray, conf: float) -> list[dict[str, Any]]:
    """Run YOLO inference and filter bounding boxes to dynamic categories."""
    results = detector(img, conf=conf, verbose=False)
    detections = []
    if not results or len(results) == 0:
        return detections

    res = results[0]
    names = res.names
    for box in res.boxes:
        cls_id = int(box.cls[0])
        cls_name = names.get(cls_id, "")
        if cls_name.lower() in DYNAMIC_CLASSES:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            h, w = img.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            detections.append({
                "class": cls_name,
                "confidence": float(box.conf[0]),
                "bbox": (x1, y1, x2, y2),
            })
    return detections


def _detect_motion_differences(img1: np.ndarray, img2: np.ndarray) -> np.ndarray:
    """Detect moving objects by aligning frames via partial affine transform.

    Returns a uint8 mask where 255 indicates moving objects and 0 indicates background.
    """
    g1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    h, w = g1.shape

    # Fast feature matching to estimate background camera motion
    orb = cv2.ORB_create(nfeatures=500)
    kp1, des1 = orb.detectAndCompute(g1, None)
    kp2, des2 = orb.detectAndCompute(g2, None)

    if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:100]

        if len(matches) >= 8:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches])
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches])

            # Robust affine transform for global camera motion
            M, inliers = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC)
            if M is not None:
                # Warp img1 into img2's perspective
                warped_g1 = cv2.warpAffine(g1, M, (w, h))
                diff = cv2.absdiff(warped_g1, g2)
                _, thresh = cv2.threshold(diff, 35, 255, cv2.THRESH_BINARY)

                # Morphological filtering to eliminate small noise and close object bodies
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
                cleaned = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)
                cleaned = cv2.dilate(cleaned, kernel, iterations=2)
                return cleaned

    # Fallback if motion alignment failed: standard difference
    diff = cv2.absdiff(g1, g2)
    _, thresh = cv2.threshold(diff, 40, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    return cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

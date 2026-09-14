"""Monocular depth estimation service with learned models and fallback."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

logger = logging.getLogger("aerorecon.depth")


def estimate_depth_maps(
    image_paths: list[str],
    output_dir: str,
    device: Optional[str] = None,
    on_progress: Optional[Callable[[float, str], None]] = None,
) -> dict[str, Any]:
    """Estimate dense depth maps for a sequence of keyframe images.

    Saves:
      - .npy raw float32 depth arrays in `output_dir/raw`
      - .png visual colorized maps in `output_dir/visualizations`

    Returns dictionary with artifact paths and depth statistics.
    """
    out = Path(output_dir)
    raw_dir = out / "raw"
    vis_dir = out / "visualizations"
    raw_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    n_images = len(image_paths)
    if n_images == 0:
        return {
            "success": True,
            "raw_dir": str(raw_dir),
            "vis_dir": str(vis_dir),
            "manifest": [],
            "stats": {"total": 0},
        }

    # Attempt to load PyTorch depth model
    torch_model, torch_device = _init_torch_depth_model(device)

    manifest: list[dict[str, Any]] = []
    all_means: list[float] = []

    for i, img_path_str in enumerate(image_paths):
        if on_progress:
            pct = 10 + (i / max(1, n_images)) * 80
            on_progress(pct, f"Estimating depth map: {i+1}/{n_images}…")

        img_p = Path(img_path_str)
        img = cv2.imread(str(img_p))
        if img is None:
            continue

        h, w = img.shape[:2]

        if torch_model is not None:
            try:
                depth_map = _infer_torch_depth(torch_model, torch_device, img)
            except Exception as exc:
                logger.warning("Torch depth inference failed (%s); using gradient depth fallback.", exc)
                depth_map = _estimate_gradient_depth(img)
        else:
            depth_map = _estimate_gradient_depth(img)

        # Normalize depth map to metric/relative scale [0.5, 50.0] meters
        d_min = float(depth_map.min())
        d_max = float(depth_map.max())
        d_mean = float(depth_map.mean())
        d_std = float(depth_map.std())

        all_means.append(d_mean)

        # 1. Save raw float32 npy
        raw_filename = f"depth_{img_p.stem}.npy"
        raw_path = raw_dir / raw_filename
        np.save(str(raw_path), depth_map.astype(np.float32))

        # 2. Save colorized visualization PNG
        # Normalize to 0..255 for color mapping
        d_norm = ((depth_map - d_min) / (d_max - d_min + 1e-6) * 255.0).astype(np.uint8)
        color_depth = cv2.applyColorMap(d_norm, cv2.COLORMAP_TURBO)

        vis_filename = f"depth_vis_{img_p.stem}.png"
        vis_path = vis_dir / vis_filename
        cv2.imwrite(str(vis_path), color_depth)

        manifest.append({
            "image_name": img_p.name,
            "raw_path": str(raw_path),
            "vis_path": str(vis_path),
            "width": w,
            "height": h,
            "min_depth": round(d_min, 3),
            "max_depth": round(d_max, 3),
            "mean_depth": round(d_mean, 3),
            "std_depth": round(d_std, 3),
        })

    manifest_json = out / "depth_manifest.json"
    manifest_json.write_text(json.dumps(manifest, indent=2))

    if on_progress:
        on_progress(100, "Depth estimation complete.")

    stats = {
        "total_depth_maps": len(manifest),
        "mean_scene_depth": round(float(np.mean(all_means)), 3) if all_means else 0.0,
        "method": "torch_midas" if torch_model is not None else "gradient_structure_fallback",
        "device": str(torch_device) if torch_device else "cpu",
    }

    logger.info("Depth estimation complete for %d frames (method=%s)", len(manifest), stats["method"])
    return {
        "success": True,
        "raw_dir": str(raw_dir),
        "vis_dir": str(vis_dir),
        "manifest_path": str(manifest_json),
        "stats": stats,
    }


def _init_torch_depth_model(preferred_device: Optional[str] = None):
    """Load MiDaS small model if PyTorch is available, with graceful CPU fallback."""
    try:
        import torch

        if preferred_device:
            dev = torch.device(preferred_device)
        else:
            dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load lightweight MiDaS model
        model = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, verbose=False)
        model.to(dev)
        model.eval()
        logger.info("Loaded MiDaS_small depth model on %s", dev)
        return model, dev
    except Exception as exc:
        logger.info("PyTorch depth model not loaded (%s); using native structure fallback.", exc)
        return None, None


def _infer_torch_depth(model, device, img_bgr: np.ndarray) -> np.ndarray:
    """Run MiDaS inference on a single BGR image."""
    import torch

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]

    # Preprocess (resize to 256x256 multiple)
    net_w, net_h = 256, 256
    resized = cv2.resize(img_rgb, (net_w, net_h), interpolation=cv2.INTER_CUBIC)
    tensor = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    # Normalize with ImageNet mean/std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor = tensor.to(device)

    with torch.no_grad():
        prediction = model(tensor)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=(h, w),
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    output = prediction.cpu().numpy()
    # Invert so higher value = farther distance, lower = closer
    d_min, d_max = output.min(), output.max()
    norm_disp = (output - d_min) / (d_max - d_min + 1e-6)
    # Convert disparity to depth: depth = 1 / (disp + epsilon)
    depth = 1.0 / (norm_disp * 0.95 + 0.05)
    return depth.astype(np.float32)


def _estimate_gradient_depth(img_bgr: np.ndarray) -> np.ndarray:
    """Structure and multi-scale texture-preserving depth approximation.

    Uses luminance gradient decay, bilateral smoothing, and vertical aerial
    depth prior (higher in image / horizons tend to be further, ground near center is closer).
    """
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # 1. Base aerial perspective gradient (vertical horizon-to-nadir ramp)
    y_coords = np.linspace(0.8, 0.2, h)[:, None]
    aerial_prior = np.repeat(y_coords, w, axis=1)

    # 2. Multi-scale texture frequency (smoother areas = sky or distant ground; high-freq = close objects)
    blur1 = cv2.GaussianBlur(gray, (5, 5), 1.0)
    blur2 = cv2.GaussianBlur(gray, (21, 21), 5.0)
    dog = cv2.absdiff(blur1, blur2).astype(np.float32) / 255.0

    # 3. Luminance-contrast component
    lum = gray.astype(np.float32) / 255.0

    # Composite depth estimate
    raw_depth = aerial_prior * 0.5 + (1.0 - dog) * 0.3 + lum * 0.2
    # Smooth with bilateral filter to preserve building / ground edges
    depth_smooth = cv2.bilateralFilter(raw_depth.astype(np.float32), d=9, sigmaColor=0.15, sigmaSpace=9)

    # Scale to nominal drone altitude range: 10m to 60m
    scaled = 10.0 + depth_smooth * 50.0
    return scaled.astype(np.float32)

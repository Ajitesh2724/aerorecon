"""Keyframe selection — intelligent frame sampling from drone video."""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np

from ..config import settings

logger = logging.getLogger("aerorecon.keyframes")


def select_keyframes(
    frames_meta: list[dict],
    target_count: Optional[int] = None,
    min_blur: Optional[float] = None,
    min_motion: Optional[float] = None,
    scene_change_threshold: float = 0.7,
) -> list[dict]:
    """Select high-quality keyframes from extracted frame metadata.

    Strategy:
      1. Filter out blurry (unusable) frames.
      2. Detect scene-change boundaries using histogram correlation.
      3. Score remaining frames by a composite quality metric.
      4. Sample uniformly across the video, boosting frames near scene changes.
      5. Return at most `target_count` keyframes.

    Each returned dict is an augmented copy of the input with added fields:
      - is_keyframe: True
      - keyframe_rank: 0-based index among selected keyframes
      - selection_reason: why this frame was selected
    """
    target = target_count or settings.TARGET_KEYFRAME_COUNT
    blur_threshold = min_blur if min_blur is not None else settings.KEYFRAME_MIN_BLUR_SCORE
    motion_threshold = min_motion if min_motion is not None else settings.KEYFRAME_MIN_MOTION_SCORE

    if not frames_meta:
        logger.warning("No frames to select from.")
        return []

    # ── Step 1: Filter usable frames ─────────────────────────────
    usable = [f for f in frames_meta if f.get("blur_score", 0) >= blur_threshold]
    if not usable:
        logger.warning("All frames below blur threshold (%.1f). Using all frames.", blur_threshold)
        usable = list(frames_meta)

    logger.info(
        "Usable frames: %d / %d (blur threshold %.1f)",
        len(usable), len(frames_meta), blur_threshold,
    )

    # If we already have fewer than target, return all
    if len(usable) <= target:
        for i, f in enumerate(usable):
            f["is_keyframe"] = True
            f["keyframe_rank"] = i
            f["selection_reason"] = "all_usable"
        return usable

    # ── Step 2: Detect scene changes ─────────────────────────────
    scene_change_indices: set[int] = set()
    for f in usable:
        hist_diff = f.get("histogram_diff", 1.0)
        if hist_diff < scene_change_threshold:
            scene_change_indices.add(f["index"])

    # ── Step 3: Score each frame ─────────────────────────────────
    blur_scores = np.array([f.get("blur_score", 0) for f in usable], dtype=np.float64)
    motion_scores = np.array([f.get("motion_score", 0) for f in usable], dtype=np.float64)

    # Normalise
    def _norm(arr: np.ndarray) -> np.ndarray:
        mn, mx = arr.min(), arr.max()
        return (arr - mn) / (mx - mn + 1e-8)

    blur_norm = _norm(blur_scores)
    motion_norm = _norm(motion_scores)

    # Composite: prefer sharp frames with moderate motion
    # Moderate motion = not too static (boring) and not too fast (blur risk)
    motion_ideal = np.abs(motion_norm - 0.4)  # penalise extremes
    composite = 0.6 * blur_norm + 0.3 * (1 - motion_ideal) + 0.1

    # Boost scene-change frames
    for i, f in enumerate(usable):
        if f["index"] in scene_change_indices:
            composite[i] += 0.3

    # ── Step 4: Uniform sampling with quality bias ───────────────
    n = len(usable)
    # Divide into `target` windows and pick the best frame in each
    window_size = max(1, n // target)
    selected: list[dict] = []
    selected_indices: set[int] = set()

    for win_start in range(0, n, window_size):
        win_end = min(win_start + window_size, n)
        if len(selected) >= target:
            break

        # Pick highest-scoring frame in this window
        window_scores = composite[win_start:win_end]
        best_local = int(np.argmax(window_scores))
        best_idx = win_start + best_local

        if best_idx not in selected_indices:
            selected_indices.add(best_idx)
            frame = dict(usable[best_idx])
            frame["is_keyframe"] = True
            frame["keyframe_rank"] = len(selected)
            frame["selection_reason"] = (
                "scene_change" if usable[best_idx]["index"] in scene_change_indices
                else "quality_score"
            )
            frame["composite_score"] = round(float(composite[best_idx]), 3)
            selected.append(frame)

    # Ensure first and last frames are included
    _ensure_boundary(usable, selected, 0, "first_frame", composite)
    _ensure_boundary(usable, selected, -1, "last_frame", composite)

    # Sort by original frame index
    selected.sort(key=lambda f: f["index"])
    for i, f in enumerate(selected):
        f["keyframe_rank"] = i

    logger.info(
        "Selected %d keyframes (%d scene changes detected)",
        len(selected), len(scene_change_indices),
    )
    return selected


def _ensure_boundary(
    usable: list[dict],
    selected: list[dict],
    pos: int,
    reason: str,
    scores: np.ndarray,
) -> None:
    """Make sure the frame at position `pos` is in the selection."""
    target_idx = usable[pos]["index"]
    if any(f["index"] == target_idx for f in selected):
        return
    frame = dict(usable[pos])
    frame["is_keyframe"] = True
    frame["keyframe_rank"] = len(selected)
    frame["selection_reason"] = reason
    frame["composite_score"] = round(float(scores[pos]), 3)
    selected.append(frame)


def summarise_keyframes(keyframes: list[dict]) -> dict:
    """Return a summary dict for logging/UI."""
    if not keyframes:
        return {"count": 0}
    blur_scores = [f.get("blur_score", 0) for f in keyframes]
    motion_scores = [f.get("motion_score", 0) for f in keyframes]
    reasons = {}
    for f in keyframes:
        r = f.get("selection_reason", "unknown")
        reasons[r] = reasons.get(r, 0) + 1

    return {
        "count": len(keyframes),
        "blur_mean": round(float(np.mean(blur_scores)), 1),
        "blur_min": round(float(np.min(blur_scores)), 1),
        "blur_max": round(float(np.max(blur_scores)), 1),
        "motion_mean": round(float(np.mean(motion_scores)), 4),
        "selection_reasons": reasons,
        "first_frame_idx": keyframes[0]["index"],
        "last_frame_idx": keyframes[-1]["index"],
    }

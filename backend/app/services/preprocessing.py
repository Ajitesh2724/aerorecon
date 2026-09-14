"""Video preprocessing — validation, frame extraction, and quality analysis."""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from ..config import settings

logger = logging.getLogger("aerorecon.preprocess")


class VideoInfo:
    """Container for video metadata extracted via OpenCV."""

    def __init__(
        self,
        path: str,
        width: int,
        height: int,
        fps: float,
        frame_count: int,
        duration_s: float,
        codec: str,
        file_size_mb: float,
    ):
        self.path = path
        self.width = width
        self.height = height
        self.fps = fps
        self.frame_count = frame_count
        self.duration_s = duration_s
        self.codec = codec
        self.file_size_mb = file_size_mb

    def to_dict(self) -> dict:
        return {
            "filename": Path(self.path).name,
            "format": Path(self.path).suffix.lstrip("."),
            "codec": self.codec,
            "width": self.width,
            "height": self.height,
            "fps": round(self.fps, 2),
            "duration_s": round(self.duration_s, 2),
            "frame_count": self.frame_count,
            "file_size_mb": round(self.file_size_mb, 2),
        }


def validate_video(video_path: str) -> VideoInfo:
    """Open a video file, validate it, and return metadata.

    Raises ValueError if the file is unreadable or outside limits.
    """
    path = Path(video_path)
    if not path.exists():
        raise ValueError(f"Video file not found: {video_path}")

    file_size_mb = path.stat().st_size / (1024 * 1024)
    if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise ValueError(
            f"Video ({file_size_mb:.0f} MB) exceeds {settings.MAX_UPLOAD_SIZE_MB} MB limit."
        )

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    try:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fourcc = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join(chr((fourcc >> (8 * i)) & 0xFF) for i in range(4))

        if fps <= 0 or frame_count <= 0:
            raise ValueError("Video has invalid FPS or frame count.")

        duration_s = frame_count / fps
        if duration_s > settings.MAX_VIDEO_DURATION_S:
            raise ValueError(
                f"Video duration ({duration_s:.0f}s) exceeds {settings.MAX_VIDEO_DURATION_S}s limit."
            )

        # Verify we can actually read a frame
        ok, frame = cap.read()
        if not ok or frame is None:
            raise ValueError("Cannot read frames from the video.")

        info = VideoInfo(
            path=video_path,
            width=width,
            height=height,
            fps=fps,
            frame_count=frame_count,
            duration_s=duration_s,
            codec=codec,
            file_size_mb=file_size_mb,
        )
        logger.info(
            "Video validated: %dx%d @ %.1f fps, %d frames, %.1fs, %.1f MB",
            width, height, fps, frame_count, duration_s, file_size_mb,
        )
        return info

    finally:
        cap.release()


def compute_blur_score(frame: np.ndarray) -> float:
    """Laplacian variance — higher = sharper."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_motion_score(prev_gray: np.ndarray, curr_gray: np.ndarray) -> float:
    """Normalised absolute frame difference — higher = more motion."""
    diff = cv2.absdiff(prev_gray, curr_gray)
    return float(diff.mean()) / 255.0


def compute_histogram_diff(prev_frame: np.ndarray, curr_frame: np.ndarray) -> float:
    """Histogram correlation — lower = bigger scene change."""
    def _hist(frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        h = cv2.calcHist([hsv], [0, 1], None, [50, 60], [0, 180, 0, 256])
        cv2.normalize(h, h)
        return h

    h1, h2 = _hist(prev_frame), _hist(curr_frame)
    return float(cv2.compareHist(h1, h2, cv2.HISTCMP_CORREL))


def extract_all_frames(
    video_path: str,
    output_dir: str,
    max_frames: Optional[int] = None,
    resize_width: Optional[int] = None,
) -> list[dict]:
    """Extract all frames (or up to max_frames) and compute quality metrics.

    Returns a list of dicts:
      { index, path, blur_score, motion_score, histogram_diff, is_usable }
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    frames_meta: list[dict] = []
    prev_gray: Optional[np.ndarray] = None
    prev_frame: Optional[np.ndarray] = None
    idx = 0

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if max_frames and idx >= max_frames:
                break

            # Optional resize for speed
            if resize_width and frame.shape[1] > resize_width:
                scale = resize_width / frame.shape[1]
                frame = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            blur = compute_blur_score(frame)

            motion = 0.0
            hist_diff = 1.0
            if prev_gray is not None:
                motion = compute_motion_score(prev_gray, gray)
                hist_diff = compute_histogram_diff(prev_frame, frame)  # type: ignore[arg-type]

            # Save frame image
            frame_path = out / f"frame_{idx:06d}.jpg"
            cv2.imwrite(str(frame_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

            is_usable = blur >= settings.KEYFRAME_MIN_BLUR_SCORE

            frames_meta.append({
                "index": idx,
                "path": str(frame_path),
                "blur_score": round(blur, 2),
                "motion_score": round(motion, 4),
                "histogram_diff": round(hist_diff, 4),
                "is_usable": is_usable,
            })

            prev_gray = gray
            prev_frame = frame.copy()
            idx += 1

            if idx % 100 == 0:
                logger.debug("Extracted %d frames…", idx)

    finally:
        cap.release()

    logger.info("Extracted %d frames to %s", len(frames_meta), output_dir)
    return frames_meta


def generate_thumbnail(video_path: str, output_path: str, timestamp_s: float = 1.0) -> bool:
    """Extract a single frame as a JPEG thumbnail."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False
    try:
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(timestamp_s * fps))
        ok, frame = cap.read()
        if not ok:
            return False
        # Resize to thumbnail
        h, w = frame.shape[:2]
        thumb_w = 480
        scale = thumb_w / w
        thumb = cv2.resize(frame, (thumb_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(output_path, thumb, [cv2.IMWRITE_JPEG_QUALITY, 85])
        return True
    finally:
        cap.release()

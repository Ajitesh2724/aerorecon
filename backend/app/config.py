"""AeroRecon configuration — settings, paths, and environment detection."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment variables and .env file."""

    # ── Application ──────────────────────────────────────────────
    APP_NAME: str = "AeroRecon"
    APP_VERSION: str = "0.1.0"
    APP_DESCRIPTION: str = "Single-pass drone video to 3D model reconstruction"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    # ── Paths (derived from project root) ────────────────────────
    BASE_DIR: Path = Field(
        default_factory=lambda: Path(__file__).resolve().parent.parent.parent
    )

    # ── External tool overrides ──────────────────────────────────
    COLMAP_PATH: Optional[str] = None
    FFMPEG_PATH: Optional[str] = None

    # ── Video processing ─────────────────────────────────────────
    MAX_UPLOAD_SIZE_MB: int = 2048
    MAX_VIDEO_DURATION_S: int = 600
    SUPPORTED_VIDEO_FORMATS: list[str] = [".mp4", ".avi", ".mov", ".mkv", ".webm"]
    KEYFRAME_MIN_BLUR_SCORE: float = 100.0
    KEYFRAME_MIN_MOTION_SCORE: float = 0.01
    TARGET_KEYFRAME_COUNT: int = 100

    # ── GPU ───────────────────────────────────────────────────────
    USE_GPU: bool = True
    GPU_DEVICE: int = 0

    # ── AI Models ─────────────────────────────────────────────────
    DEPTH_MODEL: str = "depth-anything-v2-vits"
    FLOW_MODEL: str = "raft-small"
    DETECTION_MODEL: str = "yolov8n-seg"

    # ── CORS ──────────────────────────────────────────────────────
    CORS_ORIGINS: list[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
    ]

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}

    # ── Derived paths (computed, not from env) ───────────────────

    @property
    def DATA_DIR(self) -> Path:
        return self.BASE_DIR / "data"

    @property
    def UPLOAD_DIR(self) -> Path:
        return self.DATA_DIR / "uploads"

    @property
    def JOBS_DIR(self) -> Path:
        return self.DATA_DIR / "jobs"

    @property
    def MODELS_DIR(self) -> Path:
        return self.DATA_DIR / "models"

    @property
    def DB_PATH(self) -> Path:
        return self.DATA_DIR / "aerorecon.db"

    # ── Helpers ───────────────────────────────────────────────────

    def ensure_directories(self) -> None:
        """Create required data directories if they do not exist."""
        for directory in [self.DATA_DIR, self.UPLOAD_DIR, self.JOBS_DIR, self.MODELS_DIR]:
            directory.mkdir(parents=True, exist_ok=True)

    def detect_tools(self) -> dict:
        """Probe the runtime for available external tools and GPU."""
        result: dict = {
            "colmap": self.COLMAP_PATH or shutil.which("colmap"),
            "ffmpeg": self.FFMPEG_PATH or shutil.which("ffmpeg"),
            "gpu_available": False,
            "gpu_name": None,
            "cuda_version": None,
            "gpu_memory_mb": None,
        }
        try:
            import torch  # noqa: F811

            result["gpu_available"] = torch.cuda.is_available() and self.USE_GPU
            if result["gpu_available"]:
                result["gpu_name"] = torch.cuda.get_device_name(self.GPU_DEVICE)
                result["cuda_version"] = torch.version.cuda
                props = torch.cuda.get_device_properties(self.GPU_DEVICE)
                result["gpu_memory_mb"] = round(props.total_mem / (1024**2))
        except ImportError:
            pass
        return result


settings = Settings()

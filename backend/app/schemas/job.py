"""Pydantic schemas for the AeroRecon REST API."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────


class JobStatus(str, Enum):
    PENDING = "pending"
    QUEUED = "queued"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class ProcessingStage(str, Enum):
    PREPROCESSING = "preprocessing"
    MASKING = "masking"
    SFM = "sfm"
    DEPTH = "depth"
    OPTICAL_FLOW = "optical_flow"
    DENSE_RECONSTRUCTION = "dense_reconstruction"
    GEOREFERENCING = "georeferencing"
    CONFIDENCE = "confidence"
    EXPORT = "export"


class ConfidenceTier(str, Enum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNSEEN = "unseen"


# ── Nested models ────────────────────────────────────────────────


class VideoMetadata(BaseModel):
    filename: str
    format: str = ""
    codec: Optional[str] = None
    width: int = 0
    height: int = 0
    fps: float = 0.0
    duration_s: float = 0.0
    frame_count: int = 0
    file_size_mb: float = 0.0


class StageProgress(BaseModel):
    status: str = "pending"  # pending | running | completed | failed | skipped
    progress: float = 0.0  # 0–100
    message: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


# ── API responses ────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    gpu_name: Optional[str] = None
    gpu_memory_mb: Optional[int] = None
    cuda_version: Optional[str] = None
    colmap_available: bool
    ffmpeg_available: bool


class JobResponse(BaseModel):
    id: str
    name: str
    status: str
    current_stage: Optional[str] = None
    progress: float = 0.0
    created_at: str
    updated_at: str
    completed_at: Optional[str] = None
    video_filename: Optional[str] = None
    video_metadata: Dict[str, Any] = Field(default_factory=dict)
    stage_progress: Dict[str, Any] = Field(default_factory=dict)
    artifacts: Dict[str, str] = Field(default_factory=dict)
    errors: List[str] = Field(default_factory=list)
    processing_duration_s: Optional[float] = None
    confidence_summary: Dict[str, Any] = Field(default_factory=dict)
    reconstruction_stats: Dict[str, Any] = Field(default_factory=dict)
    georef_status: str = "unavailable"


class JobListResponse(BaseModel):
    jobs: List[JobResponse]
    total: int


class ConfidenceResponse(BaseModel):
    job_id: str
    overall_tier: str
    distribution: Dict[str, float] = Field(default_factory=dict)
    factors: Dict[str, Any] = Field(default_factory=dict)
    is_estimated: bool = True


class ProcessResponse(BaseModel):
    job_id: str
    message: str
    status: str


class ErrorResponse(BaseModel):
    detail: str

"""AeroRecon REST + WebSocket API routes."""

from __future__ import annotations

import logging
import shutil
import uuid
from pathlib import Path
from typing import Optional

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
    WebSocket,
    WebSocketDisconnect,
)
from fastapi.responses import FileResponse

from ..config import settings
from ..models.job import JobCRUD
from ..schemas.job import (
    ConfidenceResponse,
    HealthResponse,
    JobListResponse,
    JobResponse,
    ProcessResponse,
)
from .websocket import manager

logger = logging.getLogger("aerorecon.api")
router = APIRouter()


# ═══════════════════════════════════════════════════════════════════
#  Health
# ═══════════════════════════════════════════════════════════════════


@router.get("/health", response_model=HealthResponse, tags=["system"])
async def health_check():
    """Return system health and tool availability."""
    tools = settings.detect_tools()
    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        gpu_available=tools["gpu_available"],
        gpu_name=tools.get("gpu_name"),
        gpu_memory_mb=tools.get("gpu_memory_mb"),
        cuda_version=tools.get("cuda_version"),
        colmap_available=bool(tools.get("colmap")),
        ffmpeg_available=bool(tools.get("ffmpeg")),
    )


# ═══════════════════════════════════════════════════════════════════
#  Jobs CRUD
# ═══════════════════════════════════════════════════════════════════


@router.post("/api/jobs", response_model=JobResponse, tags=["jobs"])
async def create_job(
    video: UploadFile = File(...),
    name: Optional[str] = Form(None),
    telemetry: Optional[UploadFile] = File(None),
    camera_intrinsics: Optional[str] = Form(None),
):
    """Upload a drone video and create a reconstruction job."""
    if not video.filename:
        raise HTTPException(status_code=400, detail="No video file provided.")

    ext = Path(video.filename).suffix.lower()
    if ext not in settings.SUPPORTED_VIDEO_FORMATS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Accepted: {settings.SUPPORTED_VIDEO_FORMATS}",
        )

    job_id = uuid.uuid4().hex[:12]
    job_name = name or f"Reconstruction {job_id[:8]}"

    # Persist video to job directory
    job_dir = settings.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    video_path = job_dir / f"input{ext}"
    try:
        content = await video.read()
        file_size_mb = len(content) / (1024 * 1024)
        if file_size_mb > settings.MAX_UPLOAD_SIZE_MB:
            shutil.rmtree(job_dir, ignore_errors=True)
            raise HTTPException(
                status_code=413,
                detail=f"Video exceeds maximum size ({settings.MAX_UPLOAD_SIZE_MB} MB).",
            )
        video_path.write_bytes(content)
    except HTTPException:
        raise
    except Exception as exc:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}")

    # Save telemetry if provided
    telemetry_path_str: Optional[str] = None
    if telemetry and telemetry.filename:
        tel_ext = Path(telemetry.filename).suffix.lower()
        tel_path = job_dir / f"telemetry{tel_ext}"
        tel_path.write_bytes(await telemetry.read())
        telemetry_path_str = str(tel_path)

    # Create database record
    job = await JobCRUD.create(
        job_id=job_id,
        name=job_name,
        video_filename=video.filename,
        video_path=str(video_path),
        telemetry_path=telemetry_path_str,
        camera_intrinsics=camera_intrinsics,
    )

    logger.info("Job %s created — video: %s (%.1f MB)", job_id, video.filename, file_size_mb)
    return JobResponse(**job)


@router.get("/api/jobs", response_model=JobListResponse, tags=["jobs"])
async def list_jobs():
    """List all reconstruction jobs (newest first)."""
    jobs = await JobCRUD.list_all()
    total = await JobCRUD.count()
    return JobListResponse(
        jobs=[JobResponse(**j) for j in jobs],
        total=total,
    )


@router.get("/api/jobs/{job_id}", response_model=JobResponse, tags=["jobs"])
async def get_job(job_id: str):
    """Get details of a specific job."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return JobResponse(**job)


@router.delete("/api/jobs/{job_id}", tags=["jobs"])
async def delete_job(job_id: str):
    """Delete a job and its associated files."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    # Remove files
    job_dir = settings.JOBS_DIR / job_id
    if job_dir.exists():
        shutil.rmtree(job_dir, ignore_errors=True)

    await JobCRUD.delete(job_id)
    return {"detail": "Job deleted.", "job_id": job_id}


# ═══════════════════════════════════════════════════════════════════
#  Processing
# ═══════════════════════════════════════════════════════════════════


async def _run_pipeline(job_id: str) -> None:
    """Background pipeline execution — delegates to the pipeline orchestrator."""
    from ..workers.pipeline import run_pipeline

    try:
        await run_pipeline(job_id)
    except Exception as exc:
        logger.exception("Pipeline failed for job %s", job_id)
        await JobCRUD.update(job_id, status="failed")
        await JobCRUD.append_error(job_id, str(exc))
        await manager.broadcast_job_update(job_id, {"status": "failed", "error": str(exc)})


@router.post("/api/jobs/{job_id}/process", response_model=ProcessResponse, tags=["processing"])
async def start_processing(job_id: str, background_tasks: BackgroundTasks):
    """Kick off the reconstruction pipeline for a job."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    if job["status"] not in ("pending", "failed"):
        raise HTTPException(
            status_code=409,
            detail=f"Job is already '{job['status']}'. Only pending or failed jobs can be (re)processed.",
        )

    await JobCRUD.update(job_id, status="queued")
    background_tasks.add_task(_run_pipeline, job_id)

    return ProcessResponse(
        job_id=job_id,
        message="Reconstruction pipeline queued.",
        status="queued",
    )


@router.get("/api/jobs/{job_id}/status", tags=["processing"])
async def get_job_status(job_id: str):
    """Lightweight status poll (stage + progress only)."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "current_stage": job.get("current_stage"),
        "progress": job.get("progress", 0),
        "stage_progress": job.get("stage_progress", {}),
    }


# ═══════════════════════════════════════════════════════════════════
#  Results / Confidence / Download
# ═══════════════════════════════════════════════════════════════════


@router.get("/api/jobs/{job_id}/results", tags=["results"])
async def get_job_results(job_id: str):
    """Return reconstruction results and artifact paths."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    return {
        "job_id": job_id,
        "status": job["status"],
        "artifacts": job.get("artifacts", {}),
        "video_metadata": job.get("video_metadata", {}),
        "confidence_summary": job.get("confidence_summary", {}),
        "georef_status": job.get("georef_status", "unavailable"),
        "processing_duration_s": job.get("processing_duration_s"),
    }


@router.get("/api/jobs/{job_id}/confidence", response_model=ConfidenceResponse, tags=["results"])
async def get_job_confidence(job_id: str):
    """Return per-job confidence scoring breakdown."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")
    summary = job.get("confidence_summary", {})
    return ConfidenceResponse(
        job_id=job_id,
        overall_tier=summary.get("overall_tier", "unseen"),
        distribution=summary.get("distribution", {}),
        factors=summary.get("factors", {}),
        is_estimated=summary.get("is_estimated", True),
    )


@router.get("/api/jobs/{job_id}/download/{artifact}", tags=["results"])
async def download_artifact(job_id: str, artifact: str):
    """Download a named artifact file from a completed job."""
    job = await JobCRUD.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job not found.")

    artifacts: dict = job.get("artifacts", {})
    if artifact not in artifacts:
        available = list(artifacts.keys()) or ["(none)"]
        raise HTTPException(
            status_code=404,
            detail=f"Artifact '{artifact}' not found. Available: {available}",
        )

    file_path = Path(artifacts[artifact])
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="Artifact file missing from disk.")

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type="application/octet-stream",
    )


# ═══════════════════════════════════════════════════════════════════
#  WebSocket
# ═══════════════════════════════════════════════════════════════════


@router.websocket("/ws")
async def websocket_global(websocket: WebSocket):
    """Global WebSocket — receive updates for all jobs."""
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)


@router.websocket("/ws/{job_id}")
async def websocket_job(websocket: WebSocket, job_id: str):
    """Job-specific WebSocket — receive updates for one job."""
    await manager.connect(websocket, job_id=job_id)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        manager.disconnect(websocket, job_id=job_id)

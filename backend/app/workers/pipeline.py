"""Pipeline orchestrator — chains processing stages for a reconstruction job."""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Coroutine, Optional

from ..config import settings
from ..models.job import JobCRUD
from ..api.websocket import manager

logger = logging.getLogger("aerorecon.pipeline")


# ── Stage result ──────────────────────────────────────────────────


class StageResult:
    """Outcome of a single pipeline stage."""

    def __init__(
        self,
        success: bool,
        message: str = "",
        artifacts: Optional[dict[str, str]] = None,
        metadata: Optional[dict[str, Any]] = None,
    ):
        self.success = success
        self.message = message
        self.artifacts = artifacts or {}
        self.metadata = metadata or {}


# ── Pipeline runner ───────────────────────────────────────────────


STAGE_ORDER = [
    "preprocessing",
    "masking",
    "sfm",
    "depth",
    "optical_flow",
    "dense_reconstruction",
    "georeferencing",
    "confidence",
    "export",
]


async def run_pipeline(job_id: str) -> None:
    """Execute all pipeline stages sequentially for a job.

    Each stage is imported and invoked dynamically. Stages that are not
    yet implemented are skipped with a log message.
    """
    start = time.time()
    logger.info("Pipeline started for job %s", job_id)

    job = await JobCRUD.get(job_id)
    if not job:
        logger.error("Job %s not found.", job_id)
        return

    job_dir = settings.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Initialise stage progress
    stage_progress = {
        stage: {"status": "pending", "progress": 0.0}
        for stage in STAGE_ORDER
    }
    all_artifacts: dict[str, str] = {}

    await JobCRUD.update(
        job_id,
        status="processing",
        stage_progress=stage_progress,
    )

    for i, stage_name in enumerate(STAGE_ORDER):
        overall_progress = (i / len(STAGE_ORDER)) * 100

        # Update current stage
        stage_progress[stage_name]["status"] = "running"
        stage_progress[stage_name]["progress"] = 0.0
        await JobCRUD.update(
            job_id,
            current_stage=stage_name,
            progress=overall_progress,
            stage_progress=stage_progress,
        )
        await _broadcast(job_id, stage_name, "running", overall_progress)
        await JobCRUD.append_log(job_id, f"Stage '{stage_name}' starting…")

        try:
            result = await _execute_stage(job_id, stage_name, job, job_dir)

            if result is None:
                # Stage not implemented — skip
                stage_progress[stage_name]["status"] = "skipped"
                stage_progress[stage_name]["message"] = "Not yet implemented"
                await JobCRUD.append_log(job_id, f"Stage '{stage_name}' skipped (not implemented).")
                continue

            if result.success:
                stage_progress[stage_name]["status"] = "completed"
                stage_progress[stage_name]["progress"] = 100.0
                stage_progress[stage_name]["message"] = result.message
                all_artifacts.update(result.artifacts)
                await JobCRUD.append_log(job_id, f"Stage '{stage_name}' completed: {result.message}")
            else:
                stage_progress[stage_name]["status"] = "failed"
                stage_progress[stage_name]["message"] = result.message
                await JobCRUD.append_error(job_id, f"[{stage_name}] {result.message}")
                await JobCRUD.append_log(job_id, f"Stage '{stage_name}' failed: {result.message}")

                # Non-critical stages can be skipped; critical ones abort
                if stage_name in ("preprocessing",):
                    # Preprocessing failure is fatal
                    await _fail_job(job_id, stage_progress, all_artifacts, start)
                    return
                # Other stages: continue with reduced quality
                logger.warning("Stage '%s' failed, continuing pipeline.", stage_name)

        except Exception as exc:
            logger.exception("Unhandled error in stage '%s' for job %s", stage_name, job_id)
            stage_progress[stage_name]["status"] = "failed"
            stage_progress[stage_name]["message"] = str(exc)
            await JobCRUD.append_error(job_id, f"[{stage_name}] {exc}")

            if stage_name in ("preprocessing",):
                await _fail_job(job_id, stage_progress, all_artifacts, start)
                return

        # Refresh job data between stages (other stages may need it)
        job = await JobCRUD.get(job_id) or job

    # ── Pipeline complete ─────────────────────────────────────
    elapsed = time.time() - start
    await JobCRUD.update(
        job_id,
        status="completed",
        progress=100.0,
        processing_duration_s=round(elapsed, 1),
        stage_progress=stage_progress,
        artifacts=all_artifacts,
        completed_at=datetime.now(timezone.utc).isoformat(),
    )
    await _broadcast(job_id, "complete", "completed", 100.0)
    await JobCRUD.append_log(job_id, f"Pipeline completed in {elapsed:.1f}s.")
    logger.info("Pipeline finished for job %s in %.1fs", job_id, elapsed)


async def _execute_stage(
    job_id: str,
    stage_name: str,
    job: dict,
    job_dir: Path,
) -> Optional[StageResult]:
    """Dispatch to the appropriate stage implementation.

    Returns None if the stage is not yet implemented.
    """
    if stage_name == "preprocessing":
        return await _run_preprocessing(job_id, job, job_dir)
    if stage_name == "masking":
        return await _run_masking(job_id, job, job_dir)
    if stage_name == "sfm":
        return await _run_sfm(job_id, job, job_dir)
    if stage_name == "depth":
        return await _run_depth(job_id, job, job_dir)
    if stage_name == "optical_flow":
        return await _run_optical_flow(job_id, job, job_dir)
    if stage_name == "dense_reconstruction":
        return await _run_dense_reconstruction(job_id, job, job_dir)
    # Stages added in later milestones return None (skipped)
    return None


# ── Stage implementations ─────────────────────────────────────────


async def _run_preprocessing(job_id: str, job: dict, job_dir: Path) -> StageResult:
    """Stage 1: Validate video, extract frames, select keyframes."""
    from ..services.preprocessing import validate_video, extract_all_frames, generate_thumbnail
    from ..services.keyframes import select_keyframes, summarise_keyframes
    from ..services.metadata import parse_telemetry, parse_camera_intrinsics, telemetry_summary

    video_path = job.get("video_path")
    if not video_path or not Path(video_path).exists():
        return StageResult(False, "Video file not found.")

    # Validate
    try:
        info = validate_video(video_path)
    except ValueError as exc:
        return StageResult(False, str(exc))

    await JobCRUD.update(job_id, video_metadata=info.to_dict())

    # Thumbnail
    thumb_path = str(job_dir / "thumbnail.jpg")
    generate_thumbnail(video_path, thumb_path)

    # Progress callback helper
    async def _update_progress(pct: float, msg: str = "") -> None:
        await JobCRUD.update(
            job_id,
            stage_progress={
                **((await JobCRUD.get(job_id)) or {}).get("stage_progress", {}),
                "preprocessing": {"status": "running", "progress": pct, "message": msg},
            },
        )
        await _broadcast(job_id, "preprocessing", "running", pct)

    await _update_progress(20, "Extracting frames…")

    # Extract frames
    frames_dir = str(job_dir / "frames")
    frames_meta = extract_all_frames(
        video_path, frames_dir,
        max_frames=2000,  # cap for large videos
        resize_width=1280 if info.width > 1920 else None,
    )

    await _update_progress(60, "Selecting keyframes…")

    # Select keyframes
    keyframes = select_keyframes(frames_meta)
    kf_summary = summarise_keyframes(keyframes)

    # Save keyframe list
    kf_json_path = job_dir / "keyframes.json"
    import json as _json
    kf_json_path.write_text(_json.dumps(keyframes, indent=2))

    await _update_progress(80, "Parsing metadata…")

    # Parse telemetry
    tel_records = None
    tel_sum: dict = {"available": False}
    if job.get("telemetry_path"):
        tel_records = parse_telemetry(job["telemetry_path"])
        if tel_records:
            tel_sum = telemetry_summary(tel_records)

    # Parse camera intrinsics
    cam_intrinsics = None
    if job.get("camera_intrinsics"):
        cam_intrinsics = parse_camera_intrinsics(job["camera_intrinsics"])

    await _update_progress(100, "Preprocessing complete.")

    artifacts = {
        "thumbnail": thumb_path,
        "keyframes_json": str(kf_json_path),
    }

    # Store parsed metadata in the job
    await JobCRUD.update(
        job_id,
        video_metadata={
            **info.to_dict(),
            "keyframes": kf_summary,
            "telemetry": tel_sum,
            "camera_intrinsics": cam_intrinsics,
        },
    )

    msg = (
        f"{info.frame_count} frames, {kf_summary['count']} keyframes selected, "
        f"{info.duration_s:.1f}s duration"
    )
    return StageResult(True, msg, artifacts)


async def _run_sfm(job_id: str, job: dict, job_dir: Path) -> StageResult:
    """Stage 2: Structure-from-Motion (COLMAP with OpenCV fallback)."""
    import json as _json
    import asyncio
    from ..services.sfm import get_colmap_path, run_colmap_sfm, run_opencv_sfm

    # Progress helper
    async def _update_progress(pct: float, msg: str = "") -> None:
        await JobCRUD.update(
            job_id,
            stage_progress={
                **((await JobCRUD.get(job_id)) or {}).get("stage_progress", {}),
                "sfm": {"status": "running", "progress": pct, "message": msg},
            },
        )
        await _broadcast(job_id, "sfm", "running", pct)

    await _update_progress(5, "Preparing keyframe images for SfM…")

    # Locate keyframe images or extracted frames
    kf_json_path = job_dir / "keyframes.json"
    sfm_img_dir = job_dir / "sfm_images"
    sfm_img_dir.mkdir(parents=True, exist_ok=True)

    image_paths = []
    if kf_json_path.exists():
        try:
            kfs = _json.loads(kf_json_path.read_text())
            for kf in kfs:
                p = kf.get("path")
                if p and Path(p).exists():
                    image_paths.append(Path(p))
        except Exception as exc:
            logger.warning("Could not read keyframes.json: %s", exc)

    if not image_paths:
        frames_dir = job_dir / "frames"
        if frames_dir.exists():
            image_paths = sorted(
                [f for f in frames_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )

    if len(image_paths) < 2:
        return StageResult(False, f"Not enough frames for SfM ({len(image_paths)} found).")

    # Copy keyframes into sfm_images directory if not already there
    for src in image_paths:
        dst = sfm_img_dir / src.name
        if not dst.exists():
            try:
                import shutil
                shutil.copy2(src, dst)
            except Exception:
                pass

    # Retrieve camera intrinsics if available
    cam_intrinsics = None
    vmeta = job.get("video_metadata") or {}
    if isinstance(vmeta, dict):
        cam_intrinsics = vmeta.get("camera_intrinsics")
    if not cam_intrinsics and job.get("camera_intrinsics"):
        from ..services.metadata import parse_camera_intrinsics
        cam_intrinsics = parse_camera_intrinsics(job["camera_intrinsics"])

    sfm_ws = job_dir / "sfm"
    sfm_ws.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()

    def sync_progress(pct: float, msg: str) -> None:
        asyncio.run_coroutine_threadsafe(_update_progress(pct, msg), loop)

    use_colmap = get_colmap_path() is not None
    sfm_result = None

    if use_colmap:
        try:
            logger.info("Executing COLMAP SfM for job %s", job_id)
            sfm_result = await loop.run_in_executor(
                None,
                lambda: run_colmap_sfm(
                    str(sfm_img_dir),
                    str(sfm_ws),
                    camera_intrinsics=cam_intrinsics,
                    on_progress=sync_progress,
                ),
            )
        except Exception as exc:
            logger.warning("COLMAP failed (%s). Falling back to OpenCV SfM.", exc)
            sfm_result = None

    if not sfm_result:
        logger.info("Executing OpenCV SfM fallback for job %s", job_id)
        await _update_progress(15, "Running OpenCV feature matching & SfM fallback…")
        try:
            sfm_result = await loop.run_in_executor(
                None,
                lambda: run_opencv_sfm(
                    str(sfm_img_dir),
                    str(sfm_ws),
                    camera_intrinsics=cam_intrinsics,
                    on_progress=sync_progress,
                ),
            )
        except Exception as exc:
            return StageResult(False, f"SfM reconstruction failed: {str(exc)}")

    await _update_progress(100, "SfM complete.")

    artifacts = {
        "sparse_ply": sfm_result["sparse_ply"],
        "poses_json": sfm_result["poses_json"],
    }
    stats = sfm_result.get("stats", {})

    # Update job with reconstruction stats
    await JobCRUD.update(
        job_id,
        reconstruction_stats={
            **((await JobCRUD.get(job_id)) or {}).get("reconstruction_stats", {}),
            "sfm": stats,
            "sparse_points": stats.get("sparse_points", 0),
            "registered_images": stats.get("registered_images", sfm_result.get("n_poses", 0)),
        },
    )

    summary_msg = (
        f"SfM recovered {stats.get('sparse_points', 0)} points from "
        f"{stats.get('registered_images', sfm_result.get('n_poses', 0))} views "
        f"(method: {stats.get('method', 'colmap')})"
    )
    return StageResult(True, summary_msg, artifacts)


def _get_job_image_paths(job_dir: Path) -> list[str]:
    """Retrieve keyframe images or fallback to all extracted frames."""
    import json as _json
    kf_json_path = job_dir / "keyframes.json"
    paths: list[str] = []
    if kf_json_path.exists():
        try:
            kfs = _json.loads(kf_json_path.read_text())
            for kf in kfs:
                p = kf.get("path")
                if p and Path(p).exists():
                    paths.append(str(Path(p)))
        except Exception as exc:
            logger.warning("Could not read keyframes.json: %s", exc)

    if not paths:
        frames_dir = job_dir / "frames"
        if frames_dir.exists():
            paths = sorted(
                [str(f) for f in frames_dir.iterdir() if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
            )
    return paths


async def _run_masking(job_id: str, job: dict, job_dir: Path) -> StageResult:
    """Stage 2: Dynamic object masking."""
    import asyncio
    from ..services.masking import generate_dynamic_masks

    async def _update_progress(pct: float, msg: str = "") -> None:
        await JobCRUD.update(
            job_id,
            stage_progress={
                **((await JobCRUD.get(job_id)) or {}).get("stage_progress", {}),
                "masking": {"status": "running", "progress": pct, "message": msg},
            },
        )
        await _broadcast(job_id, "masking", "running", pct)

    image_paths = _get_job_image_paths(job_dir)
    if not image_paths:
        return StageResult(False, "No frames available for dynamic masking.")

    await _update_progress(5, "Analyzing dynamic objects & transients…")

    loop = asyncio.get_running_loop()
    def sync_progress(pct: float, msg: str) -> None:
        asyncio.run_coroutine_threadsafe(_update_progress(pct, msg), loop)

    mask_res = await loop.run_in_executor(
        None,
        lambda: generate_dynamic_masks(
            image_paths,
            str(job_dir / "masking"),
            on_progress=sync_progress,
        ),
    )

    await _update_progress(100, "Masking complete.")
    stats = mask_res.get("stats", {})
    artifacts = {
        "masks_manifest": mask_res["manifest_path"],
    }
    msg = (
        f"Analyzed {stats.get('total_frames', 0)} frames. "
        f"{stats.get('frames_with_dynamic_objects', 0)} dynamic objects isolated "
        f"(method: {stats.get('method', 'motion_compensation')})"
    )
    return StageResult(True, msg, artifacts)


async def _run_depth(job_id: str, job: dict, job_dir: Path) -> StageResult:
    """Stage 4: Monocular Depth Estimation."""
    import asyncio
    from ..services.depth import estimate_depth_maps

    async def _update_progress(pct: float, msg: str = "") -> None:
        await JobCRUD.update(
            job_id,
            stage_progress={
                **((await JobCRUD.get(job_id)) or {}).get("stage_progress", {}),
                "depth": {"status": "running", "progress": pct, "message": msg},
            },
        )
        await _broadcast(job_id, "depth", "running", pct)

    image_paths = _get_job_image_paths(job_dir)
    if not image_paths:
        return StageResult(False, "No frames available for depth estimation.")

    await _update_progress(5, "Initializing depth estimation engine…")

    loop = asyncio.get_running_loop()
    def sync_progress(pct: float, msg: str) -> None:
        asyncio.run_coroutine_threadsafe(_update_progress(pct, msg), loop)

    depth_res = await loop.run_in_executor(
        None,
        lambda: estimate_depth_maps(
            image_paths,
            str(job_dir / "depth"),
            on_progress=sync_progress,
        ),
    )

    await _update_progress(100, "Depth estimation complete.")
    stats = depth_res.get("stats", {})
    artifacts = {
        "depth_manifest": depth_res["manifest_path"],
    }
    msg = (
        f"Generated {stats.get('total_depth_maps', 0)} dense depth maps "
        f"(mean depth: {stats.get('mean_scene_depth', 0.0)}m, method: {stats.get('method', 'unknown')})"
    )
    return StageResult(True, msg, artifacts)


async def _run_optical_flow(job_id: str, job: dict, job_dir: Path) -> StageResult:
    """Stage 5: Continuous Optical Flow and Motion Priors."""
    import asyncio
    from ..services.optical_flow import compute_sequence_optical_flow

    async def _update_progress(pct: float, msg: str = "") -> None:
        await JobCRUD.update(
            job_id,
            stage_progress={
                **((await JobCRUD.get(job_id)) or {}).get("stage_progress", {}),
                "optical_flow": {"status": "running", "progress": pct, "message": msg},
            },
        )
        await _broadcast(job_id, "optical_flow", "running", pct)

    image_paths = _get_job_image_paths(job_dir)
    if len(image_paths) < 2:
        return StageResult(False, f"Not enough frames for optical flow ({len(image_paths)} found).")

    await _update_progress(5, "Estimating forward-backward optical flow fields…")

    loop = asyncio.get_running_loop()
    def sync_progress(pct: float, msg: str) -> None:
        asyncio.run_coroutine_threadsafe(_update_progress(pct, msg), loop)

    flow_res = await loop.run_in_executor(
        None,
        lambda: compute_sequence_optical_flow(
            image_paths,
            str(job_dir / "flow"),
            on_progress=sync_progress,
        ),
    )

    await _update_progress(100, "Optical flow complete.")
    stats = flow_res.get("stats", {})
    artifacts = {
        "flow_manifest": flow_res["manifest_path"],
    }
    msg = (
        f"Computed {stats.get('total_pairs', 0)} flow fields "
        f"(mean velocity: {stats.get('mean_flow_velocity_px', 0.0)} px/frame, "
        f"consistency: {round(stats.get('mean_consistency_rate', 0.0) * 100, 1)}%)"
    )
    return StageResult(True, msg, artifacts)


async def _run_dense_reconstruction(job_id: str, job: dict, job_dir: Path) -> StageResult:
    """Stage 6: Dense 3D Point Cloud, Gaussian Splats, and Surface Mesh."""
    import asyncio
    from ..services.dense_reconstruction import run_dense_reconstruction
    from ..services.gaussian_splatting import export_gaussian_splats
    from ..services.mesh import generate_surface_mesh

    async def _update_progress(pct: float, msg: str = "") -> None:
        await JobCRUD.update(
            job_id,
            stage_progress={
                **((await JobCRUD.get(job_id)) or {}).get("stage_progress", {}),
                "dense_reconstruction": {"status": "running", "progress": pct, "message": msg},
            },
        )
        await _broadcast(job_id, "dense_reconstruction", "running", pct)

    # Locate poses JSON
    poses_json = job_dir / "sfm" / "camera_poses.json"
    if not poses_json.exists():
        poses_json = job_dir / "camera_poses.json"
    if not poses_json.exists():
        return StageResult(False, "Camera poses not found. SfM stage must succeed first.")

    depth_dir = job_dir / "depth" / "raw"
    if not depth_dir.exists():
        return StageResult(False, "Depth maps not found. Depth stage must succeed first.")

    image_dir = job_dir / "sfm_images"
    if not image_dir.exists():
        image_dir = job_dir / "frames"

    masks_dir = job_dir / "masking" / "masks"

    # Intrinsics
    cam_intrinsics = None
    vmeta = job.get("video_metadata") or {}
    if isinstance(vmeta, dict):
        cam_intrinsics = vmeta.get("camera_intrinsics")

    await _update_progress(5, "Fusing multi-view depth and back-projecting points…")

    dense_out = job_dir / "dense" / "dense.ply"
    dense_out.parent.mkdir(parents=True, exist_ok=True)

    loop = asyncio.get_running_loop()
    def sync_progress(pct: float, msg: str) -> None:
        asyncio.run_coroutine_threadsafe(_update_progress(pct * 0.5, msg), loop)

    dense_res = await loop.run_in_executor(
        None,
        lambda: run_dense_reconstruction(
            poses_json_path=str(poses_json),
            depth_dir=str(depth_dir),
            image_dir=str(image_dir),
            output_ply_path=str(dense_out),
            masks_dir=str(masks_dir) if masks_dir.exists() else None,
            camera_intrinsics=cam_intrinsics,
            on_progress=sync_progress,
        ),
    )

    artifacts = {
        "dense_ply": dense_res["dense_ply"],
    }
    stats = dense_res.get("stats", {})

    # Helper to parse points and colors from ASCII PLY for splats and mesh
    def _parse_ply_data(ply_path: Path):
        pts, cols = [], []
        with open(ply_path, "r") as f:
            in_header = True
            for line in f:
                if in_header:
                    if line.strip() == "end_header":
                        in_header = False
                    continue
                parts = line.strip().split()
                if len(parts) >= 6:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    cols.append([int(parts[3]), int(parts[4]), int(parts[5])])
                elif len(parts) >= 3:
                    pts.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    cols.append([180, 180, 180])
        if not pts:
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)
        return np.array(pts, dtype=np.float32), np.array(cols, dtype=np.uint8)

    pts, cols = await loop.run_in_executor(None, lambda: _parse_ply_data(dense_out))

    # Generate 3D Gaussian Splats
    if len(pts) > 0:
        await _update_progress(55, "Generating 3D Gaussian Splatting representation…")
        splat_out = job_dir / "splat" / "gaussian_splat.ply"
        def splat_prog(pct: float, msg: str) -> None:
            asyncio.run_coroutine_threadsafe(_update_progress(50 + pct * 0.2, msg), loop)

        splat_res = await loop.run_in_executor(
            None,
            lambda: export_gaussian_splats(
                pts, cols, str(splat_out), on_progress=splat_prog
            ),
        )
        artifacts["gaussian_splat"] = splat_res["splat_ply"]
        stats["num_gaussians"] = splat_res["stats"].get("num_gaussians", 0)

    # Generate Surface Mesh
    if len(pts) >= 3:
        await _update_progress(75, "Generating continuous 3D surface mesh…")
        mesh_out = job_dir / "mesh" / "model.obj"
        def mesh_prog(pct: float, msg: str) -> None:
            asyncio.run_coroutine_threadsafe(_update_progress(70 + pct * 0.25, msg), loop)

        mesh_res = await loop.run_in_executor(
            None,
            lambda: generate_surface_mesh(
                pts, cols, str(mesh_out), on_progress=mesh_prog
            ),
        )
        artifacts["mesh_obj"] = mesh_res["mesh_obj"]
        stats["face_count"] = mesh_res["stats"].get("face_count", 0)

    await _update_progress(100, "Dense 3D reconstruction complete.")

    # Update job with reconstruction stats
    await JobCRUD.update(
        job_id,
        reconstruction_stats={
            **((await JobCRUD.get(job_id)) or {}).get("reconstruction_stats", {}),
            "dense_points": stats.get("dense_points_count", 0),
            "num_gaussians": stats.get("num_gaussians", 0),
            "face_count": stats.get("face_count", 0),
        },
    )

    msg = (
        f"Generated {stats.get('dense_points_count', 0):,} dense 3D points, "
        f"{stats.get('num_gaussians', 0):,} Gaussian splats, "
        f"and {stats.get('face_count', 0):,} mesh triangles"
    )
    return StageResult(True, msg, artifacts)


# ── Helpers ───────────────────────────────────────────────────────


async def _fail_job(
    job_id: str,
    stage_progress: dict,
    artifacts: dict,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time
    await JobCRUD.update(
        job_id,
        status="failed",
        processing_duration_s=round(elapsed, 1),
        stage_progress=stage_progress,
        artifacts=artifacts,
    )
    await _broadcast(job_id, "error", "failed", 0)
    logger.error("Pipeline failed for job %s after %.1fs", job_id, elapsed)


async def _broadcast(
    job_id: str,
    stage: str,
    status: str,
    progress: float,
) -> None:
    """Send a WebSocket update."""
    await manager.broadcast_job_update(job_id, {
        "status": status,
        "stage": stage,
        "progress": round(progress, 1),
    })

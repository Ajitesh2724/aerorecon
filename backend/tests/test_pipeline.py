"""Integration tests for pipeline execution: preprocessing through SfM."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.models.job import JobCRUD
from app.workers.pipeline import _run_preprocessing, _run_sfm
from app.config import settings


@pytest.fixture
def synthetic_drone_video(tmp_path: Path) -> str:
    """Create synthetic video with high texture and moving camera for SfM."""
    video_path = str(tmp_path / "drone_flight.mp4")
    width, height, fps, n_frames = 320, 240, 15, 20

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    np.random.seed(123)
    # Textured base terrain
    base = np.zeros((height + 100, width + 100, 3), dtype=np.uint8)
    for r in range(20, height + 80, 25):
        for c in range(20, width + 80, 25):
            color = (int(np.random.randint(50, 250)), int(np.random.randint(50, 250)), int(np.random.randint(50, 250)))
            cv2.rectangle(base, (c, r), (c + 18, r + 18), color, -1)
            cv2.circle(base, (c + 9, r + 9), 5, (255, 255, 255), 1)

    for i in range(n_frames):
        # Simulated drone forward movement
        y_offset = int(i * 2.5)
        x_offset = int(i * 1.5)
        frame = base[y_offset : y_offset + height, x_offset : x_offset + width].copy()
        writer.write(frame)

    writer.release()
    return video_path


@pytest.mark.asyncio
async def test_full_pipeline_sfm_flow(synthetic_drone_video: str, tmp_path: Path):
    job_id = "test-sfm-job-001"
    job_dir = settings.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create job
        job = await JobCRUD.create(
            job_id=job_id,
            name="Test Drone Flight",
            video_filename="drone_flight.mp4",
            video_path=synthetic_drone_video,
        )

        # 1. Run Preprocessing Stage
        pre_res = await _run_preprocessing(job_id, job, job_dir)
        assert pre_res.success is True
        assert "keyframes_json" in pre_res.artifacts
        assert Path(pre_res.artifacts["keyframes_json"]).exists()

        # Update job with preprocessing output
        await JobCRUD.update(
            job_id,
            artifacts=pre_res.artifacts,
        )
        updated_job = await JobCRUD.get(job_id)
        assert updated_job is not None

        # 2. Run SfM Stage
        sfm_res = await _run_sfm(job_id, updated_job, job_dir)
        assert sfm_res.success is True
        assert "sparse_ply" in sfm_res.artifacts
        assert "poses_json" in sfm_res.artifacts
        assert Path(sfm_res.artifacts["sparse_ply"]).exists()
        assert Path(sfm_res.artifacts["poses_json"]).exists()

        # Check job reconstruction stats
        final_job = await JobCRUD.get(job_id)
        assert final_job is not None
        assert "reconstruction_stats" in final_job
        assert final_job["reconstruction_stats"]["registered_images"] >= 2

        # 3. Test artifact download via API
        all_artifacts = {**pre_res.artifacts, **sfm_res.artifacts}
        await JobCRUD.update(job_id, artifacts=all_artifacts)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            # Download sparse PLY
            ply_resp = await client.get(f"/api/jobs/{job_id}/download/sparse_ply")
            assert ply_resp.status_code == 200
            assert "element vertex" in ply_resp.text

            # Download camera poses JSON
            poses_resp = await client.get(f"/api/jobs/{job_id}/download/poses_json")
            assert poses_resp.status_code == 200
            poses = poses_resp.json()
            assert isinstance(poses, list)
            assert len(poses) >= 2

    finally:
        # Clean up test files
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        await JobCRUD.delete(job_id)

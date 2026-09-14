"""End-to-end integration test running the complete 9-stage AeroRecon pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import zipfile

import cv2
import numpy as np
import pytest
from httpx import AsyncClient, ASGITransport

from app.main import app
from app.config import settings
from app.models.job import JobCRUD
from app.workers.pipeline import run_pipeline


@pytest.fixture
def synthetic_drone_dataset(tmp_path: Path):
    """Generate synthetic drone video and matching telemetry."""
    dataset_dir = tmp_path / "drone_dataset"
    dataset_dir.mkdir()

    video_path = str(dataset_dir / "drone_survey.mp4")
    telemetry_path = str(dataset_dir / "telemetry.json")

    width, height, fps, n_frames = 320, 240, 10, 16
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(video_path, fourcc, fps, (width, height))

    np.random.seed(42)
    # High-contrast geometric grid with simulated buildings/trees
    canvas = np.zeros((height + 150, width + 150, 3), dtype=np.uint8)
    for r in range(15, height + 130, 20):
        for c in range(15, width + 130, 20):
            color = (int(np.random.randint(60, 240)), int(np.random.randint(60, 240)), int(np.random.randint(60, 240)))
            cv2.rectangle(canvas, (c, r), (c + 14, r + 14), color, -1)
            cv2.circle(canvas, (c + 7, r + 7), 4, (255, 255, 255), 1)

    # Telemetry data
    lat0, lon0, alt0 = 28.6139, 77.2090, 250.0
    telemetry = []

    for i in range(n_frames):
        y_off = int(i * 3.0)
        x_off = int(i * 2.0)
        frame = canvas[y_off : y_off + height, x_off : x_off + width].copy()
        writer.write(frame)

        telemetry.append({
            "timestamp_s": float(i / fps),
            "latitude": lat0 + (i * 0.00002),
            "longitude": lon0 + (i * 0.00004),
            "altitude_m": alt0 + (i * 0.15),
            "yaw_deg": float(i * 1.2),
            "pitch_deg": -45.0,
            "roll_deg": 0.0,
        })

    writer.release()
    Path(telemetry_path).write_text(json.dumps(telemetry, indent=2))

    return {
        "video_path": video_path,
        "telemetry_path": telemetry_path,
        "telemetry": telemetry,
    }


@pytest.mark.asyncio
async def test_full_pipeline_end_to_end(synthetic_drone_dataset: dict):
    job_id = "test-e2e-drone-job-999"
    job_dir = settings.JOBS_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    try:
        # Create Job in SQLite
        job = await JobCRUD.create(
            job_id=job_id,
            name="E2E Drone Inspection Survey",
            video_filename="drone_survey.mp4",
            video_path=synthetic_drone_dataset["video_path"],
            telemetry_path=synthetic_drone_dataset["telemetry_path"],
        )
        assert job is not None
        assert job["status"] == "pending"

        # Run complete pipeline (Stages 1 through 9)
        await run_pipeline(job_id)

        # Verify job completed successfully
        completed_job = await JobCRUD.get(job_id)
        assert completed_job is not None
        assert completed_job["status"] == "completed"
        assert completed_job["progress"] == 100.0

        stage_progress = completed_job.get("stage_progress", {})
        assert stage_progress.get("preprocessing", {}).get("status") == "completed"
        assert stage_progress.get("sfm", {}).get("status") == "completed"
        assert stage_progress.get("dense_reconstruction", {}).get("status") == "completed"
        assert stage_progress.get("georeferencing", {}).get("status") == "completed"
        assert stage_progress.get("confidence", {}).get("status") == "completed"
        assert stage_progress.get("export", {}).get("status") == "completed"

        # Verify artifacts
        artifacts = completed_job.get("artifacts", {})
        assert "sparse_ply" in artifacts
        assert "dense_ply" in artifacts
        assert "gaussian_splats" in artifacts
        assert "mesh_obj" in artifacts
        assert "flight_path" in artifacts
        assert "confidence_report" in artifacts
        assert "export_package" in artifacts

        # Verify files on disk
        for key in ["sparse_ply", "dense_ply", "gaussian_splats", "mesh_obj", "export_package"]:
            assert Path(artifacts[key]).exists(), f"Artifact {key} does not exist on disk"

        # Check export zip contents
        zip_p = Path(artifacts["export_package"])
        with zipfile.ZipFile(zip_p, "r") as zf:
            namelist = zf.namelist()
            assert "README.txt" in namelist
            assert any("dense.ply" in n for n in namelist)
            assert any("gaussian_splat.ply" in n for n in namelist)
            assert any("model.obj" in n for n in namelist)

        # Check API endpoints for this completed job
        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as client:
            res = await client.get(f"/api/jobs/{job_id}/results")
            assert res.status_code == 200
            data = res.json()
            assert data["status"] == "completed"
            assert "artifacts" in data
            assert data["georef_status"] in ("aligned", "uncalibrated")

            conf_res = await client.get(f"/api/jobs/{job_id}/confidence")
            assert conf_res.status_code == 200
            conf_data = conf_res.json()
            assert conf_data["overall_tier"] in ("high", "medium", "low")
            assert "distribution" in conf_data

            dl_res = await client.get(f"/api/jobs/{job_id}/download/export_package")
            assert dl_res.status_code == 200
            assert len(dl_res.content) > 1000

    finally:
        import shutil
        shutil.rmtree(job_dir, ignore_errors=True)
        await JobCRUD.delete(job_id)

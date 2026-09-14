"""API endpoint tests for AeroRecon backend."""

from __future__ import annotations

import io
import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.models import database as db_module
from app.models.database import init_db, close_db, CREATE_JOBS_TABLE
from app.config import settings

import aiosqlite


@pytest.fixture(autouse=True)
async def fresh_db(tmp_path):
    """Use a temporary database for each test so tests are fully isolated."""
    test_db_path = tmp_path / "test.db"
    # Monkey-patch settings to use temp path
    original_db = settings.DB_PATH
    object.__setattr__(settings, '_db_path_override', test_db_path)

    # Close any prior connection and open a fresh one
    await close_db()
    db_module._db = await aiosqlite.connect(str(test_db_path))
    db_module._db.row_factory = aiosqlite.Row
    await db_module._db.execute(CREATE_JOBS_TABLE)
    await db_module._db.commit()

    yield

    await close_db()
    object.__setattr__(settings, '_db_path_override', None)


@pytest.mark.anyio
async def test_health_endpoint():
    """GET /health returns healthy status and tool info."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert "gpu_available" in data
    assert "colmap_available" in data
    assert "ffmpeg_available" in data


@pytest.mark.anyio
async def test_list_jobs_empty():
    """GET /api/jobs with no jobs returns an empty list."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/jobs")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["jobs"] == []


@pytest.mark.anyio
async def test_create_and_get_job():
    """POST /api/jobs creates a job; GET /api/jobs/{id} retrieves it."""
    # Create a tiny test video file (just bytes — validation is minimal in M1)
    video_content = b"\x00" * 1024
    files = {"video": ("test_drone.mp4", io.BytesIO(video_content), "video/mp4")}
    data = {"name": "Test Flight"}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        # Create
        resp = await client.post("/api/jobs", files=files, data=data)
        assert resp.status_code == 200
        job = resp.json()
        assert job["name"] == "Test Flight"
        assert job["status"] == "pending"
        job_id = job["id"]

        # Get
        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 200
        assert resp.json()["id"] == job_id

        # List
        resp = await client.get("/api/jobs")
        assert resp.status_code == 200
        assert resp.json()["total"] == 1


@pytest.mark.anyio
async def test_create_job_unsupported_format():
    """POST /api/jobs rejects unsupported video formats."""
    files = {"video": ("test.txt", io.BytesIO(b"not a video"), "text/plain")}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/jobs", files=files)
    assert resp.status_code == 400
    assert "Unsupported" in resp.json()["detail"]


@pytest.mark.anyio
async def test_delete_job():
    """DELETE /api/jobs/{id} removes the job."""
    video_content = b"\x00" * 512
    files = {"video": ("flight.mp4", io.BytesIO(video_content), "video/mp4")}

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/jobs", files=files)
        job_id = resp.json()["id"]

        resp = await client.delete(f"/api/jobs/{job_id}")
        assert resp.status_code == 200

        resp = await client.get(f"/api/jobs/{job_id}")
        assert resp.status_code == 404


@pytest.mark.anyio
async def test_get_nonexistent_job():
    """GET /api/jobs/{id} returns 404 for missing job."""
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/jobs/nonexistent")
    assert resp.status_code == 404


@pytest.mark.anyio
async def test_process_pending_job():
    """POST /api/jobs/{id}/process accepts a pending job."""
    files = {"video": ("drone.mp4", io.BytesIO(b"\x00" * 512), "video/mp4")}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/jobs", files=files)
        job_id = resp.json()["id"]

        resp = await client.post(f"/api/jobs/{job_id}/process")
        assert resp.status_code == 200
        assert resp.json()["status"] == "queued"


@pytest.mark.anyio
async def test_job_status_endpoint():
    """GET /api/jobs/{id}/status returns lightweight status."""
    files = {"video": ("vid.mp4", io.BytesIO(b"\x00" * 256), "video/mp4")}
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/jobs", files=files)
        job_id = resp.json()["id"]

        resp = await client.get(f"/api/jobs/{job_id}/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["job_id"] == job_id
        assert data["status"] == "pending"

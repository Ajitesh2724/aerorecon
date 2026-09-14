"""AeroRecon FastAPI application entry point."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from .config import settings
from .api.routes import router
from .models.database import init_db, close_db

logging.basicConfig(
    level=logging.DEBUG if settings.DEBUG else logging.INFO,
    format="%(asctime)s | %(name)-18s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("aerorecon")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    # ── Startup ──────────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info(f"  {settings.APP_NAME} v{settings.APP_VERSION}")
    logger.info("  Single-pass drone video → 3D model reconstruction")
    logger.info("=" * 60)

    settings.ensure_directories()
    await init_db()

    tools = settings.detect_tools()
    logger.info(
        "COLMAP : %s",
        f"✓ {tools['colmap']}" if tools["colmap"] else "✗  not found",
    )
    logger.info(
        "FFmpeg : %s",
        f"✓ {tools['ffmpeg']}" if tools["ffmpeg"] else "✗  not found",
    )
    gpu_msg = "✗  not available"
    if tools["gpu_available"]:
        gpu_msg = f"✓ {tools['gpu_name']} ({tools['gpu_memory_mb']} MB, CUDA {tools['cuda_version']})"
    logger.info("GPU    : %s", gpu_msg)
    logger.info("-" * 60)

    yield

    # ── Shutdown ─────────────────────────────────────────────────
    await close_db()
    logger.info("AeroRecon stopped.")


# ── Application instance ─────────────────────────────────────────

app = FastAPI(
    title=settings.APP_NAME,
    description=settings.APP_DESCRIPTION,
    version=settings.APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API routes
app.include_router(router)

# Serve job output files (point clouds, thumbnails, etc.)
# The directory is created at startup via ensure_directories()
app.mount(
    "/static/jobs",
    StaticFiles(directory=str(settings.JOBS_DIR), check_dir=False),
    name="job_files",
)

"""SQLite database management for AeroRecon job persistence."""

from __future__ import annotations

import logging

import aiosqlite

from ..config import settings

logger = logging.getLogger("aerorecon.db")

_db: aiosqlite.Connection | None = None

# ── Schema ────────────────────────────────────────────────────────

CREATE_JOBS_TABLE = """
CREATE TABLE IF NOT EXISTS jobs (
    id                  TEXT PRIMARY KEY,
    name                TEXT NOT NULL,
    status              TEXT NOT NULL DEFAULT 'pending',
    current_stage       TEXT,
    progress            REAL NOT NULL DEFAULT 0.0,
    created_at          TEXT NOT NULL,
    updated_at          TEXT NOT NULL,
    completed_at        TEXT,
    video_filename      TEXT,
    video_path          TEXT,
    telemetry_path      TEXT,
    camera_intrinsics   TEXT,
    video_metadata      TEXT NOT NULL DEFAULT '{}',
    stage_progress      TEXT NOT NULL DEFAULT '{}',
    artifacts           TEXT NOT NULL DEFAULT '{}',
    errors              TEXT NOT NULL DEFAULT '[]',
    logs                TEXT NOT NULL DEFAULT '',
    processing_duration_s REAL,
    confidence_summary  TEXT NOT NULL DEFAULT '{}',
    georef_status       TEXT NOT NULL DEFAULT 'unavailable',
    reconstruction_stats TEXT NOT NULL DEFAULT '{}'
);
"""


async def init_db() -> None:
    """Create the database and tables if they do not exist."""
    global _db
    db_path = settings.DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Opening database at %s", db_path)
    _db = await aiosqlite.connect(str(db_path))
    _db.row_factory = aiosqlite.Row
    await _db.execute(CREATE_JOBS_TABLE)
    try:
        await _db.execute(
            "ALTER TABLE jobs ADD COLUMN reconstruction_stats TEXT NOT NULL DEFAULT '{}'"
        )
    except Exception:
        pass  # Column already exists
    await _db.commit()


async def get_db() -> aiosqlite.Connection:
    """Return the active database connection, initialising if needed."""
    global _db
    if _db is None:
        await init_db()
    assert _db is not None
    return _db


async def close_db() -> None:
    """Close the database connection."""
    global _db
    if _db is not None:
        await _db.close()
        _db = None
        logger.info("Database closed.")

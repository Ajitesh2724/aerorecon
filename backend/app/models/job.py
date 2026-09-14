"""Job CRUD operations against the SQLite database."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .database import get_db

logger = logging.getLogger("aerorecon.job")

# JSON-serialised columns that need parsing on read
_JSON_FIELDS = frozenset(
    {"video_metadata", "stage_progress", "artifacts", "errors", "confidence_summary", "reconstruction_stats"}
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_dict(row) -> dict[str, Any]:
    """Convert an aiosqlite Row to a plain dict, parsing JSON columns."""
    data = dict(row)
    for field in _JSON_FIELDS:
        raw = data.get(field)
        if isinstance(raw, str):
            try:
                data[field] = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                data[field] = {} if field != "errors" else []
    return data


class JobCRUD:
    """Async CRUD helpers for the *jobs* table."""

    # ── Create ────────────────────────────────────────────────────

    @staticmethod
    async def create(
        *,
        job_id: str,
        name: str,
        video_filename: Optional[str] = None,
        video_path: Optional[str] = None,
        telemetry_path: Optional[str] = None,
        camera_intrinsics: Optional[str] = None,
    ) -> dict[str, Any]:
        db = await get_db()
        now = _now_iso()
        await db.execute(
            """
            INSERT INTO jobs
                (id, name, status, created_at, updated_at,
                 video_filename, video_path, telemetry_path, camera_intrinsics)
            VALUES (?, ?, 'pending', ?, ?, ?, ?, ?, ?)
            """,
            (
                job_id, name, now, now,
                video_filename, video_path, telemetry_path, camera_intrinsics,
            ),
        )
        await db.commit()
        logger.info("Created job %s (%s)", job_id, name)
        return await JobCRUD.get(job_id)  # type: ignore[return-value]

    # ── Read ──────────────────────────────────────────────────────

    @staticmethod
    async def get(job_id: str) -> Optional[dict[str, Any]]:
        db = await get_db()
        cursor = await db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = await cursor.fetchone()
        return _row_to_dict(row) if row else None

    @staticmethod
    async def list_all(
        limit: int = 50,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        db = await get_db()
        cursor = await db.execute(
            "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (limit, offset),
        )
        rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    @staticmethod
    async def count() -> int:
        db = await get_db()
        cursor = await db.execute("SELECT COUNT(*) FROM jobs")
        row = await cursor.fetchone()
        return row[0] if row else 0

    # ── Update ────────────────────────────────────────────────────

    @staticmethod
    async def update(job_id: str, **fields: Any) -> Optional[dict[str, Any]]:
        """Update arbitrary columns on a job.

        JSON-serialisable values for JSON columns are automatically dumped.
        ``updated_at`` is always refreshed.
        """
        if not fields:
            return await JobCRUD.get(job_id)

        fields["updated_at"] = _now_iso()

        # Serialise dicts/lists destined for JSON columns
        for key, value in fields.items():
            if key in _JSON_FIELDS and not isinstance(value, str):
                fields[key] = json.dumps(value)

        set_clause = ", ".join(f"{k} = ?" for k in fields)
        values = list(fields.values()) + [job_id]

        db = await get_db()
        await db.execute(
            f"UPDATE jobs SET {set_clause} WHERE id = ?",
            values,
        )
        await db.commit()
        return await JobCRUD.get(job_id)

    @staticmethod
    async def append_log(job_id: str, message: str) -> None:
        """Append a timestamped line to the job's log buffer."""
        db = await get_db()
        ts = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{ts}] {message}\n"
        await db.execute(
            "UPDATE jobs SET logs = logs || ?, updated_at = ? WHERE id = ?",
            (line, _now_iso(), job_id),
        )
        await db.commit()

    @staticmethod
    async def append_error(job_id: str, error: str) -> None:
        """Add an error string to the job's error list."""
        db = await get_db()
        cursor = await db.execute(
            "SELECT errors FROM jobs WHERE id = ?", (job_id,)
        )
        row = await cursor.fetchone()
        if row is None:
            return
        errors = json.loads(row["errors"]) if isinstance(row["errors"], str) else row["errors"]
        errors.append(error)
        await db.execute(
            "UPDATE jobs SET errors = ?, updated_at = ? WHERE id = ?",
            (json.dumps(errors), _now_iso(), job_id),
        )
        await db.commit()

    # ── Delete ────────────────────────────────────────────────────

    @staticmethod
    async def delete(job_id: str) -> bool:
        db = await get_db()
        cursor = await db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        await db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted job %s", job_id)
        return deleted

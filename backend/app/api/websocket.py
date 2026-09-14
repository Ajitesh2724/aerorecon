"""WebSocket connection manager for real-time job progress updates."""

from __future__ import annotations

import json
import logging
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger("aerorecon.ws")


class ConnectionManager:
    """Manages WebSocket connections and broadcasts job updates.

    Clients can subscribe globally (receive all job updates) or
    to a specific job_id.
    """

    def __init__(self) -> None:
        # job_id → set of sockets watching that specific job
        self._job_connections: dict[str, set[WebSocket]] = {}
        # sockets watching all jobs
        self._global_connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket, job_id: str | None = None) -> None:
        await websocket.accept()
        if job_id:
            self._job_connections.setdefault(job_id, set()).add(websocket)
            logger.debug("WS client subscribed to job %s", job_id)
        else:
            self._global_connections.add(websocket)
            logger.debug("WS client subscribed globally")

    def disconnect(self, websocket: WebSocket, job_id: str | None = None) -> None:
        if job_id and job_id in self._job_connections:
            self._job_connections[job_id].discard(websocket)
            if not self._job_connections[job_id]:
                del self._job_connections[job_id]
        self._global_connections.discard(websocket)

    async def broadcast_job_update(
        self,
        job_id: str,
        data: dict[str, Any],
    ) -> None:
        """Send a job update to all relevant subscribers."""
        message = json.dumps({"type": "job_update", "job_id": job_id, **data})

        # Job-specific subscribers
        for ws in list(self._job_connections.get(job_id, [])):
            try:
                await ws.send_text(message)
            except Exception:
                self._job_connections.get(job_id, set()).discard(ws)

        # Global subscribers
        for ws in list(self._global_connections):
            try:
                await ws.send_text(message)
            except Exception:
                self._global_connections.discard(ws)

    @property
    def connection_count(self) -> int:
        job_conns = sum(len(s) for s in self._job_connections.values())
        return job_conns + len(self._global_connections)


# Singleton instance used across the application
manager = ConnectionManager()

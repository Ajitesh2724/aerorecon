/* ── API service layer ─────────────────────────────────────────── */

import type { Job, JobListResponse, HealthInfo } from '../types/job';

const API_BASE = '';

async function request<T>(url: string, options?: RequestInit): Promise<T> {
  const resp = await fetch(`${API_BASE}${url}`, {
    ...options,
    headers: { ...options?.headers },
  });
  if (!resp.ok) {
    const body = await resp.json().catch(() => ({ detail: resp.statusText }));
    throw new Error(body.detail || `API error ${resp.status}`);
  }
  return resp.json();
}

export const api = {
  /** System health and tool availability. */
  health: () => request<HealthInfo>('/health'),

  /** List all reconstruction jobs. */
  listJobs: () => request<JobListResponse>('/api/jobs'),

  /** Get a single job by ID. */
  getJob: (id: string) => request<Job>(`/api/jobs/${id}`),

  /** Upload a video and create a new job. */
  createJob: async (
    video: File,
    name?: string,
    telemetry?: File,
    cameraIntrinsics?: string,
  ): Promise<Job> => {
    const form = new FormData();
    form.append('video', video);
    if (name) form.append('name', name);
    if (telemetry) form.append('telemetry', telemetry);
    if (cameraIntrinsics) form.append('camera_intrinsics', cameraIntrinsics);

    return request<Job>('/api/jobs', { method: 'POST', body: form });
  },

  /** Start the reconstruction pipeline for a job. */
  processJob: (id: string) =>
    request<{ job_id: string; status: string; message: string }>(
      `/api/jobs/${id}/process`,
      { method: 'POST' },
    ),

  /** Lightweight status poll. */
  getJobStatus: (id: string) =>
    request<{ job_id: string; status: string; current_stage?: string; progress: number }>(
      `/api/jobs/${id}/status`,
    ),

  /** Full results and artifact paths. */
  getJobResults: (id: string) =>
    request<Record<string, unknown>>(`/api/jobs/${id}/results`),

  /** Confidence scoring breakdown. */
  getJobConfidence: (id: string) =>
    request<Record<string, unknown>>(`/api/jobs/${id}/confidence`),

  /** Delete a job and its files. */
  deleteJob: (id: string) =>
    request<{ detail: string }>(`/api/jobs/${id}`, { method: 'DELETE' }),

  /** Get download URL for an artifact. */
  artifactUrl: (jobId: string, artifact: string) =>
    `${API_BASE}/api/jobs/${jobId}/download/${artifact}`,
};


/* ── WebSocket helper ──────────────────────────────────────────── */

export type WSMessage = {
  type: 'job_update';
  job_id: string;
  status?: string;
  stage?: string;
  progress?: number;
  error?: string;
};

export function connectWebSocket(
  jobId?: string,
  onMessage?: (msg: WSMessage) => void,
  onClose?: () => void,
): WebSocket {
  const proto = window.location.protocol === 'https:' ? 'wss' : 'ws';
  const path = jobId ? `/ws/${jobId}` : '/ws';
  const ws = new WebSocket(`${proto}://${window.location.host}${path}`);

  ws.onmessage = (event) => {
    try {
      const data = JSON.parse(event.data) as WSMessage;
      onMessage?.(data);
    } catch {
      /* ignore non-JSON messages */
    }
  };

  ws.onclose = () => onClose?.();
  return ws;
}

/* ── Job type definitions ─────────────────────────────────────── */

export type JobStatus = 'pending' | 'queued' | 'processing' | 'completed' | 'failed' | 'cancelled';

export type ProcessingStage =
  | 'preprocessing'
  | 'masking'
  | 'sfm'
  | 'depth'
  | 'optical_flow'
  | 'dense_reconstruction'
  | 'georeferencing'
  | 'confidence'
  | 'export';

export type ConfidenceTier = 'high' | 'medium' | 'low' | 'unseen';

export interface VideoMetadata {
  filename: string;
  format: string;
  codec?: string;
  width: number;
  height: number;
  fps: number;
  duration_s: number;
  frame_count: number;
  file_size_mb: number;
}

export interface StageProgress {
  status: 'pending' | 'running' | 'completed' | 'failed' | 'skipped';
  progress: number;
  message?: string;
  started_at?: string;
  completed_at?: string;
}

export interface Job {
  id: string;
  name: string;
  status: JobStatus;
  current_stage?: string;
  progress: number;
  created_at: string;
  updated_at: string;
  completed_at?: string;
  video_filename?: string;
  video_metadata: Record<string, unknown>;
  stage_progress: Record<string, StageProgress>;
  artifacts: Record<string, string>;
  errors: string[];
  processing_duration_s?: number;
  reconstruction_stats?: Record<string, unknown>;
  confidence_summary: Record<string, unknown>;
  georef_status: string;
}

export interface HealthInfo {
  status: string;
  version: string;
  gpu_available: boolean;
  gpu_name?: string;
  gpu_memory_mb?: number;
  cuda_version?: string;
  colmap_available: boolean;
  ffmpeg_available: boolean;
}

export interface JobListResponse {
  jobs: Job[];
  total: number;
}

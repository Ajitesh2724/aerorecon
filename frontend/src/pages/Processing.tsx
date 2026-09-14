import { useEffect, useState, useRef } from 'react';
import { useParams, Link } from 'react-router-dom';
import {
  ArrowLeft,
  Play,
  Loader2,
  CheckCircle2,
  XCircle,
  Clock,
  AlertCircle,
  FileVideo,
  Box,
  Layers,
  MapPin,
  Shield,
  Download,
  Eye,
} from 'lucide-react';
import { api, connectWebSocket, type WSMessage } from '../services/api';
import type { Job } from '../types/job';
import PointCloudViewer from '../components/PointCloudViewer';

/* ── Pipeline stage definitions ────────────────────────────────── */

const STAGES = [
  { key: 'preprocessing',        label: 'Preprocessing',         icon: FileVideo },
  { key: 'masking',              label: 'Object Masking',        icon: Eye },
  { key: 'sfm',                  label: 'Structure from Motion', icon: Layers },
  { key: 'depth',                label: 'Depth Estimation',      icon: Box },
  { key: 'optical_flow',         label: 'Optical Flow',          icon: Layers },
  { key: 'dense_reconstruction', label: 'Dense Reconstruction',  icon: Box },
  { key: 'georeferencing',       label: 'Georeferencing',        icon: MapPin },
  { key: 'confidence',           label: 'Confidence Mapping',    icon: Shield },
  { key: 'export',               label: 'Export',                icon: Download },
] as const;

/* ── Processing page ───────────────────────────────────────────── */

export default function Processing() {
  const { jobId } = useParams<{ jobId: string }>();
  const [job, setJob] = useState<Job | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const wsRef = useRef<WebSocket | null>(null);

  // Fetch job and connect WebSocket
  useEffect(() => {
    if (!jobId) return;

    api.getJob(jobId)
      .then(setJob)
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));

    // WebSocket for real-time updates
    const ws = connectWebSocket(jobId, (msg: WSMessage) => {
      if (msg.job_id === jobId) {
        // Refresh job data on update
        api.getJob(jobId).then(setJob).catch(() => {});
      }
    });
    wsRef.current = ws;

    return () => {
      ws.close();
    };
  }, [jobId]);

  const handleStartProcessing = async () => {
    if (!jobId) return;
    try {
      await api.processJob(jobId);
      const updated = await api.getJob(jobId);
      setJob(updated);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Failed to start processing');
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-[var(--color-primary-400)]" />
      </div>
    );
  }

  if (!job) {
    return (
      <div className="flex h-screen flex-col items-center justify-center gap-4">
        <AlertCircle className="h-12 w-12 text-red-400" />
        <p className="text-lg text-slate-300">Job not found</p>
        <Link to="/" className="text-sm text-[var(--color-primary-400)] hover:underline">
          ← Back to Dashboard
        </Link>
      </div>
    );
  }

  const stageProgress = job.stage_progress as Record<string, { status?: string; progress?: number; message?: string }>;

  return (
    <div className="p-8 max-w-4xl mx-auto space-y-8">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <Link
            to="/"
            className="rounded-lg p-2 text-slate-400 transition-colors hover:bg-white/[0.04] hover:text-slate-200"
          >
            <ArrowLeft className="h-5 w-5" />
          </Link>
          <div>
            <h2 className="text-2xl font-bold tracking-tight text-slate-100">
              {job.name}
            </h2>
            <p className="text-sm text-slate-500">
              Job {job.id} · {job.video_filename ?? 'No video'}
            </p>
          </div>
        </div>

        {/* Action button */}
        {job.status === 'pending' && (
          <button
            onClick={handleStartProcessing}
            className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-[var(--color-primary-500)] to-[var(--color-accent-500)] px-5 py-2.5 text-sm font-semibold text-white shadow-lg transition-all hover:scale-[1.02] active:scale-[0.98]"
          >
            <Play className="h-4 w-4" />
            Start Reconstruction
          </button>
        )}
      </div>

      {/* Error */}
      {error && (
        <div className="glass-card flex items-center gap-3 border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          <AlertCircle className="h-5 w-5 shrink-0" />
          {error}
        </div>
      )}

      {/* Status card */}
      <div className="glass-card p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            <StatusIcon status={job.status} />
            <div>
              <p className="font-semibold text-slate-200 capitalize">{job.status}</p>
              <p className="text-xs text-slate-500">
                {job.current_stage ? `Stage: ${job.current_stage}` : 'Awaiting processing'}
              </p>
            </div>
          </div>
          {job.processing_duration_s && (
            <p className="text-sm text-slate-400">
              {job.processing_duration_s.toFixed(1)}s elapsed
            </p>
          )}
        </div>

        {/* Overall progress bar */}
        {(job.status === 'processing' || job.status === 'queued') && (
          <div className="mt-4">
            <div className="h-2 rounded-full bg-[var(--color-surface-600)] overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-[var(--color-primary-500)] to-[var(--color-accent-500)] transition-all duration-500 progress-active"
                style={{ width: `${Math.max(job.progress, 2)}%` }}
              />
            </div>
            <p className="mt-1 text-right text-xs text-slate-500">{job.progress.toFixed(0)}%</p>
          </div>
        )}
      </div>

      {/* Pipeline stages */}
      <div className="glass-card p-6">
        <h3 className="mb-5 text-sm font-semibold uppercase tracking-wider text-slate-400">
          Reconstruction Pipeline
        </h3>
        <div className="space-y-1">
          {STAGES.map(({ key, label, icon: Icon }, i) => {
            const stage = stageProgress[key];
            const status = stage?.status ?? 'pending';
            const progress = stage?.progress ?? 0;
            const isActive = job.current_stage === key;

            return (
              <div key={key}>
                <div className={`flex items-center gap-4 rounded-xl px-4 py-3 transition-all ${
                  isActive ? 'bg-[var(--color-primary-500)]/5 border border-[var(--color-primary-500)]/20' : ''
                }`}>
                  <StageStatusIcon status={status} isActive={isActive} />
                  <Icon className="h-4 w-4 text-slate-500" />
                  <div className="flex-1 min-w-0">
                    <p className={`text-sm font-medium ${
                      status === 'completed' ? 'text-emerald-400' :
                      isActive ? 'text-[var(--color-primary-400)]' :
                      status === 'failed' ? 'text-red-400' :
                      'text-slate-500'
                    }`}>
                      {label}
                    </p>
                    {stage?.message && (
                      <p className="text-xs text-slate-500 truncate">{stage.message}</p>
                    )}
                  </div>
                  {status === 'running' && (
                    <span className="text-xs text-slate-400">{progress.toFixed(0)}%</span>
                  )}
                </div>
                {i < STAGES.length - 1 && (
                  <div className={`ml-7 h-4 w-px ${
                    status === 'completed' ? 'bg-emerald-500/40' :
                    isActive ? 'stage-connector-active w-px' :
                    'bg-[var(--color-surface-600)]'
                  }`} />
                )}
              </div>
            );
          })}
        </div>
      </div>

      {/* Errors */}
      {job.errors.length > 0 && (
        <div className="glass-card border-red-500/20 p-6">
          <h3 className="mb-3 text-sm font-semibold uppercase tracking-wider text-red-400">
            Errors
          </h3>
          <ul className="space-y-2">
            {job.errors.map((err, i) => (
              <li key={i} className="flex items-start gap-2 text-sm text-red-300">
                <XCircle className="mt-0.5 h-4 w-4 shrink-0 text-red-500" />
                {err}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* 3D Reconstruction Viewer */}
      {(job.artifacts?.sparse_ply || job.artifacts?.dense_ply) && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              3D Reconstruction (Sparse SfM, Dense Fusion & Surface Mesh)
            </h3>
            {Boolean(job.reconstruction_stats?.sfm) && (
              <span className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-mono text-cyan-300">
                Method: {String((job.reconstruction_stats?.sfm as Record<string, unknown>)?.method || 'COLMAP')}
              </span>
            )}
          </div>

          {/* Quick Metrics Bar */}
          <div className="grid grid-cols-4 gap-3">
            <div className="glass-card p-4">
              <p className="text-xs text-slate-500">Sparse Points</p>
              <p className="text-xl font-bold text-slate-100 mt-1">
                {Number(job.reconstruction_stats?.sparse_points || 0).toLocaleString()}
              </p>
            </div>
            <div className="glass-card p-4">
              <p className="text-xs text-slate-500">Dense 3D Points</p>
              <p className="text-xl font-bold text-indigo-400 mt-1">
                {Number(job.reconstruction_stats?.dense_points || 0).toLocaleString()}
              </p>
            </div>
            <div className="glass-card p-4">
              <p className="text-xs text-slate-500">3D Gaussians</p>
              <p className="text-xl font-bold text-cyan-400 mt-1">
                {Number(job.reconstruction_stats?.num_gaussians || 0).toLocaleString()}
              </p>
            </div>
            <div className="glass-card p-4">
              <p className="text-xs text-slate-500">Mesh Triangles</p>
              <p className="text-xl font-bold text-emerald-400 mt-1">
                {Number(job.reconstruction_stats?.face_count || 0).toLocaleString()}
              </p>
            </div>
          </div>

          <PointCloudViewer
            sparsePlyUrl={job.artifacts.sparse_ply ? api.artifactUrl(job.id, 'sparse_ply') : undefined}
            densePlyUrl={job.artifacts.dense_ply ? api.artifactUrl(job.id, 'dense_ply') : undefined}
            meshUrl={job.artifacts.mesh_obj ? api.artifactUrl(job.id, 'mesh_obj') : undefined}
            posesUrl={job.artifacts.poses_json ? api.artifactUrl(job.id, 'poses_json') : undefined}
            scaleFactor={(job.reconstruction_stats?.scale_factor as number) || 1.0}
            title={`${job.name} — 3D Reconstruction & Trajectory`}
          />
        </div>
      )}

      {/* Quality & Confidence Audit Card */}
      {Boolean(job.confidence_summary && Object.keys(job.confidence_summary).length > 0) && (
        <div className="glass-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2.5">
              <Shield className="h-5 w-5 text-emerald-400" />
              <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-300">
                Reconstruction Confidence & Metric Quality Audit
              </h3>
            </div>
            <span
              className={`rounded-full px-3 py-1 text-xs font-bold uppercase tracking-wider ${
                job.confidence_summary.overall_tier === 'high'
                  ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                  : job.confidence_summary.overall_tier === 'medium'
                  ? 'bg-amber-500/20 text-amber-400 border border-amber-500/30'
                  : 'bg-red-500/20 text-red-400 border border-red-500/30'
              }`}
            >
              Tier: {String(job.confidence_summary.overall_tier || 'Medium')}
            </span>
          </div>

          {/* Stacked distribution bar */}
          {Boolean(job.confidence_summary.distribution) && (
            <div className="space-y-1.5">
              <div className="flex justify-between text-xs text-slate-400">
                <span>Confidence Distribution</span>
                <span>
                  High: {String((job.confidence_summary.distribution as Record<string, unknown>).high ?? 0)}% · Med: {String((job.confidence_summary.distribution as Record<string, unknown>).medium ?? 0)}% · Low: {String((job.confidence_summary.distribution as Record<string, unknown>).low ?? 0)}%
                </span>
              </div>
              <div className="h-3 w-full rounded-full bg-slate-800 overflow-hidden flex">
                <div
                  style={{ width: `${Number((job.confidence_summary.distribution as Record<string, unknown>).high || 0)}%` }}
                  className="bg-emerald-500 h-full transition-all"
                  title="High Confidence"
                />
                <div
                  style={{ width: `${Number((job.confidence_summary.distribution as Record<string, unknown>).medium || 0)}%` }}
                  className="bg-amber-500 h-full transition-all"
                  title="Medium Confidence"
                />
                <div
                  style={{ width: `${Number((job.confidence_summary.distribution as Record<string, unknown>).low || 0)}%` }}
                  className="bg-red-500 h-full transition-all"
                  title="Low Confidence"
                />
                <div
                  style={{ width: `${Number((job.confidence_summary.distribution as Record<string, unknown>).unseen || 0)}%` }}
                  className="bg-slate-700 h-full transition-all"
                  title="Occluded / Unseen"
                />
              </div>
            </div>
          )}

          {/* Factors Breakdown */}
          {Boolean(job.confidence_summary.factors) && (
            <div className="grid grid-cols-4 gap-3 pt-2">
              <div className="rounded-xl bg-white/[0.03] p-3 text-xs border border-white/[0.04]">
                <p className="text-slate-500">View Redundancy</p>
                <p className="text-base font-semibold text-slate-200 mt-0.5">
                  {Math.round(Number((job.confidence_summary.factors as Record<string, unknown>).view_redundancy || 0) * 100)}%
                </p>
              </div>
              <div className="rounded-xl bg-white/[0.03] p-3 text-xs border border-white/[0.04]">
                <p className="text-slate-500">Reprojection Precision</p>
                <p className="text-base font-semibold text-cyan-400 mt-0.5">
                  {Math.round(Number((job.confidence_summary.factors as Record<string, unknown>).reprojection_precision || 0) * 100)}%
                </p>
              </div>
              <div className="rounded-xl bg-white/[0.03] p-3 text-xs border border-white/[0.04]">
                <p className="text-slate-500">Motion Consistency</p>
                <p className="text-base font-semibold text-indigo-400 mt-0.5">
                  {Math.round(Number((job.confidence_summary.factors as Record<string, unknown>).flow_consistency || 0) * 100)}%
                </p>
              </div>
              <div className="rounded-xl bg-white/[0.03] p-3 text-xs border border-white/[0.04]">
                <p className="text-slate-500">Georeference Quality</p>
                <p className="text-base font-semibold text-emerald-400 mt-0.5">
                  {job.georef_status === 'aligned' ? '100% (GPS)' : 'Nominal (Default)'}
                </p>
              </div>
            </div>
          )}
        </div>
      )}

      {/* Artifacts / Downloads */}
      {Object.keys(job.artifacts).length > 0 && (
        <div className="glass-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
              Output Artifacts & Deliverables
            </h3>
            {job.artifacts.export_package && (
              <a
                href={api.artifactUrl(job.id, 'export_package')}
                className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-cyan-500 to-blue-600 px-4 py-2 text-xs font-bold text-white shadow-lg transition hover:scale-[1.02] active:scale-[0.98]"
              >
                <Download className="h-4 w-4" /> Download Complete Package (.ZIP)
              </a>
            )}
          </div>

          <div className="grid grid-cols-2 gap-3">
            {Object.entries(job.artifacts).map(([name, _path]) => (
              <a
                key={name}
                href={api.artifactUrl(job.id, name)}
                className="flex items-center gap-3 rounded-xl border border-white/[0.06] bg-[var(--color-surface-700)] px-4 py-3 text-sm text-slate-300 transition-all hover:border-[var(--color-primary-500)]/30 hover:bg-[var(--color-surface-600)]"
              >
                <Download className="h-4 w-4 text-[var(--color-primary-400)]" />
                <span className="capitalize">{name.replace(/_/g, ' ')}</span>
              </a>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}

/* ── Helper components ─────────────────────────────────────────── */

function StatusIcon({ status }: { status: string }) {
  switch (status) {
    case 'completed':
      return <CheckCircle2 className="h-8 w-8 text-emerald-400" />;
    case 'failed':
      return <XCircle className="h-8 w-8 text-red-400" />;
    case 'processing':
      return <Loader2 className="h-8 w-8 animate-spin text-[var(--color-primary-400)]" />;
    case 'queued':
      return <Clock className="h-8 w-8 text-amber-400" />;
    default:
      return <Clock className="h-8 w-8 text-slate-500" />;
  }
}

function StageStatusIcon({ status, isActive }: { status: string; isActive: boolean }) {
  if (status === 'completed') return <CheckCircle2 className="h-5 w-5 text-emerald-400" />;
  if (status === 'failed') return <XCircle className="h-5 w-5 text-red-400" />;
  if (isActive || status === 'running') return <Loader2 className="h-5 w-5 animate-spin text-[var(--color-primary-400)]" />;
  return (
    <div className="flex h-5 w-5 items-center justify-center rounded-full border border-[var(--color-surface-500)]">
      <div className="h-1.5 w-1.5 rounded-full bg-[var(--color-surface-500)]" />
    </div>
  );
}

import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  Upload,
  Box,
  Cpu,
  HardDrive,
  MonitorCheck,
  Clock,
  AlertCircle,
  CheckCircle2,
  Loader2,
  Trash2,
  ArrowRight,
  Layers,
} from 'lucide-react';
import { api } from '../services/api';
import type { Job, HealthInfo, JobStatus } from '../types/job';

/* ── Status badge ──────────────────────────────────────────────── */

function StatusBadge({ status }: { status: JobStatus }) {
  const styles: Record<string, string> = {
    pending:    'bg-slate-500/20 text-slate-400',
    queued:     'bg-amber-500/20 text-amber-400',
    processing: 'bg-blue-500/20 text-blue-400',
    completed:  'bg-emerald-500/20 text-emerald-400',
    failed:     'bg-red-500/20 text-red-400',
    cancelled:  'bg-gray-500/20 text-gray-400',
  };
  const icons: Record<string, React.ReactNode> = {
    pending:    <Clock className="h-3 w-3" />,
    queued:     <Clock className="h-3 w-3" />,
    processing: <Loader2 className="h-3 w-3 animate-spin" />,
    completed:  <CheckCircle2 className="h-3 w-3" />,
    failed:     <AlertCircle className="h-3 w-3" />,
  };

  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 text-xs font-medium ${styles[status] ?? styles.pending}`}>
      {icons[status]}
      {status}
    </span>
  );
}

/* ── Dashboard page ────────────────────────────────────────────── */

export default function Dashboard() {
  const [health, setHealth] = useState<HealthInfo | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');

  useEffect(() => {
    Promise.all([api.health(), api.listJobs()])
      .then(([h, jl]) => {
        setHealth(h);
        setJobs(jl.jobs);
      })
      .catch((e) => setError(e.message))
      .finally(() => setLoading(false));
  }, []);

  const handleDelete = async (id: string) => {
    if (!confirm('Delete this job and all its files?')) return;
    try {
      await api.deleteJob(id);
      setJobs((prev) => prev.filter((j) => j.id !== id));
    } catch (e: unknown) {
      alert(e instanceof Error ? e.message : 'Delete failed');
    }
  };

  if (loading) {
    return (
      <div className="flex h-screen items-center justify-center">
        <Loader2 className="h-8 w-8 animate-spin text-[var(--color-primary-400)]" />
      </div>
    );
  }

  return (
    <div className="p-8 space-y-8 max-w-7xl mx-auto">
      {/* Header */}
      <div className="flex items-end justify-between">
        <div>
          <h2 className="text-3xl font-bold tracking-tight gradient-text">
            Mission Control
          </h2>
          <p className="mt-1 text-sm text-slate-400">
            Drone video reconstruction dashboard
          </p>
        </div>
        <Link
          to="/upload"
          className="flex items-center gap-2 rounded-xl bg-gradient-to-r from-[var(--color-primary-500)] to-[var(--color-primary-600)] px-5 py-2.5 text-sm font-semibold text-white shadow-lg transition-all hover:shadow-[var(--color-primary-500)]/20 hover:scale-[1.02] active:scale-[0.98]"
        >
          <Upload className="h-4 w-4" />
          New Reconstruction
        </Link>
      </div>

      {error && (
        <div className="glass-card flex items-center gap-3 border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          <AlertCircle className="h-5 w-5 shrink-0" />
          {error}
        </div>
      )}

      {/* System status cards */}
      {health && (
        <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
          <StatusCard
            icon={<MonitorCheck className="h-5 w-5" />}
            label="System"
            value={health.status}
            valueClass="text-emerald-400"
            sub={`v${health.version}`}
          />
          <StatusCard
            icon={<Cpu className="h-5 w-5" />}
            label="GPU"
            value={health.gpu_available ? (health.gpu_name ?? 'Available') : 'CPU only'}
            valueClass={health.gpu_available ? 'text-emerald-400' : 'text-amber-400'}
            sub={health.gpu_available ? `CUDA ${health.cuda_version} · ${health.gpu_memory_mb} MB` : 'Models will run on CPU'}
          />
          <StatusCard
            icon={<Layers className="h-5 w-5" />}
            label="COLMAP"
            value={health.colmap_available ? 'Installed' : 'Not found'}
            valueClass={health.colmap_available ? 'text-emerald-400' : 'text-amber-400'}
            sub={health.colmap_available ? 'SfM engine ready' : 'Install for SfM pipeline'}
          />
          <StatusCard
            icon={<HardDrive className="h-5 w-5" />}
            label="FFmpeg"
            value={health.ffmpeg_available ? 'Installed' : 'Not found'}
            valueClass={health.ffmpeg_available ? 'text-emerald-400' : 'text-amber-400'}
            sub={health.ffmpeg_available ? 'Video processing ready' : 'OpenCV fallback active'}
          />
        </div>
      )}

      {/* Recent jobs */}
      <section>
        <h3 className="mb-4 text-lg font-semibold text-slate-200">
          Recent Reconstructions
        </h3>

        {jobs.length === 0 ? (
          <div className="glass-card flex flex-col items-center justify-center py-16 text-center">
            <Box className="mb-4 h-12 w-12 text-slate-600 float-animation" />
            <p className="text-lg font-medium text-slate-400">No reconstructions yet</p>
            <p className="mt-1 text-sm text-slate-500">
              Upload a drone video to start your first 3D reconstruction
            </p>
            <Link
              to="/upload"
              className="mt-6 flex items-center gap-2 rounded-xl bg-[var(--color-surface-600)] px-5 py-2.5 text-sm font-medium text-slate-300 transition-colors hover:bg-[var(--color-surface-500)]"
            >
              <Upload className="h-4 w-4" />
              Upload Video
            </Link>
          </div>
        ) : (
          <div className="space-y-3">
            {jobs.map((job) => (
              <div
                key={job.id}
                className="glass-card glass-card-hover flex items-center justify-between px-6 py-4"
              >
                <div className="flex items-center gap-4 min-w-0">
                  <div className="flex h-10 w-10 shrink-0 items-center justify-center rounded-lg bg-[var(--color-primary-500)]/10">
                    <Box className="h-5 w-5 text-[var(--color-primary-400)]" />
                  </div>
                  <div className="min-w-0">
                    <p className="truncate font-medium text-slate-200">{job.name}</p>
                    <p className="text-xs text-slate-500">
                      {job.video_filename ?? 'No video'} ·{' '}
                      {new Date(job.created_at).toLocaleDateString()}
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <StatusBadge status={job.status} />
                  <button
                    onClick={() => handleDelete(job.id)}
                    className="rounded-lg p-2 text-slate-500 transition-colors hover:bg-red-500/10 hover:text-red-400"
                    title="Delete job"
                  >
                    <Trash2 className="h-4 w-4" />
                  </button>
                  <Link
                    to={`/jobs/${job.id}`}
                    className="flex items-center gap-1 rounded-lg px-3 py-1.5 text-sm text-[var(--color-primary-400)] transition-colors hover:bg-[var(--color-primary-500)]/10"
                  >
                    View <ArrowRight className="h-3.5 w-3.5" />
                  </Link>
                </div>
              </div>
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

/* ── Status card subcomponent ──────────────────────────────────── */

function StatusCard({
  icon,
  label,
  value,
  valueClass,
  sub,
}: {
  icon: React.ReactNode;
  label: string;
  value: string;
  valueClass?: string;
  sub?: string;
}) {
  return (
    <div className="glass-card px-5 py-4">
      <div className="flex items-center gap-3">
        <div className="text-slate-500">{icon}</div>
        <span className="text-xs font-medium uppercase tracking-wider text-slate-500">{label}</span>
      </div>
      <p className={`mt-2 truncate text-base font-semibold ${valueClass ?? 'text-slate-200'}`}>
        {value}
      </p>
      {sub && <p className="mt-0.5 truncate text-xs text-slate-500">{sub}</p>}
    </div>
  );
}

import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  Upload as UploadIcon,
  FileVideo,
  MapPin,
  Camera,
  X,
  Loader2,
  CheckCircle2,
  AlertCircle,
  Info,
} from 'lucide-react';
import { api } from '../services/api';

export default function Upload() {
  const navigate = useNavigate();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [video, setVideo] = useState<File | null>(null);
  const [telemetry, setTelemetry] = useState<File | null>(null);
  const [name, setName] = useState('');
  const [cameraIntrinsics, setCameraIntrinsics] = useState('');
  const [dragOver, setDragOver] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [error, setError] = useState('');

  const SUPPORTED = ['.mp4', '.avi', '.mov', '.mkv', '.webm'];

  const handleDrop = useCallback((e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files[0];
    if (file) validateAndSetVideo(file);
  }, []);

  const validateAndSetVideo = (file: File) => {
    setError('');
    const ext = '.' + file.name.split('.').pop()?.toLowerCase();
    if (!SUPPORTED.includes(ext)) {
      setError(`Unsupported format "${ext}". Use: ${SUPPORTED.join(', ')}`);
      return;
    }
    if (file.size > 2048 * 1024 * 1024) {
      setError('File exceeds 2 GB limit.');
      return;
    }
    setVideo(file);
  };

  const handleSubmit = async () => {
    if (!video) return;
    setUploading(true);
    setError('');
    try {
      const job = await api.createJob(video, name || undefined, telemetry || undefined, cameraIntrinsics || undefined);
      navigate(`/jobs/${job.id}`);
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Upload failed');
      setUploading(false);
    }
  };

  const formatSize = (bytes: number) => {
    if (bytes > 1e9) return `${(bytes / 1e9).toFixed(1)} GB`;
    if (bytes > 1e6) return `${(bytes / 1e6).toFixed(1)} MB`;
    return `${(bytes / 1e3).toFixed(0)} KB`;
  };

  return (
    <div className="p-8 max-w-3xl mx-auto space-y-8">
      {/* Header */}
      <div>
        <h2 className="text-3xl font-bold tracking-tight gradient-text">
          New Reconstruction
        </h2>
        <p className="mt-1 text-sm text-slate-400">
          Upload a drone video to generate a 3D model
        </p>
      </div>

      {/* Error */}
      {error && (
        <div className="glass-card flex items-center gap-3 border-red-500/30 bg-red-500/10 p-4 text-sm text-red-400">
          <AlertCircle className="h-5 w-5 shrink-0" />
          {error}
          <button onClick={() => setError('')} className="ml-auto">
            <X className="h-4 w-4" />
          </button>
        </div>
      )}

      {/* ── Video drop zone ──────────────────────────────────── */}
      <div
        className={`drop-zone glass-card flex flex-col items-center justify-center py-16 cursor-pointer ${dragOver ? 'drag-over' : ''}`}
        onClick={() => fileInputRef.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept={SUPPORTED.join(',')}
          className="hidden"
          onChange={(e) => e.target.files?.[0] && validateAndSetVideo(e.target.files[0])}
        />

        {video ? (
          <div className="flex items-center gap-4">
            <div className="flex h-14 w-14 items-center justify-center rounded-2xl bg-emerald-500/10">
              <CheckCircle2 className="h-7 w-7 text-emerald-400" />
            </div>
            <div>
              <p className="font-medium text-slate-200">{video.name}</p>
              <p className="text-sm text-slate-400">{formatSize(video.size)}</p>
            </div>
            <button
              onClick={(e) => { e.stopPropagation(); setVideo(null); }}
              className="ml-4 rounded-lg p-2 text-slate-500 hover:bg-red-500/10 hover:text-red-400 transition-colors"
            >
              <X className="h-4 w-4" />
            </button>
          </div>
        ) : (
          <>
            <FileVideo className="mb-4 h-12 w-12 text-slate-500 float-animation" />
            <p className="text-lg font-medium text-slate-300">
              Drop your drone video here
            </p>
            <p className="mt-1 text-sm text-slate-500">
              or click to browse · {SUPPORTED.join(', ')} · max 2 GB
            </p>
          </>
        )}
      </div>

      {/* ── Job name ──────────────────────────────────────────── */}
      <div className="glass-card p-6 space-y-5">
        <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
          Reconstruction Details
        </h3>

        <div>
          <label className="block text-sm font-medium text-slate-300 mb-1.5">
            Project Name (optional)
          </label>
          <input
            type="text"
            value={name}
            onChange={(e) => setName(e.target.value)}
            placeholder="e.g. Site Survey — Block A"
            className="w-full rounded-xl border border-white/[0.08] bg-[var(--color-surface-700)] px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600 focus:border-[var(--color-primary-500)] focus:outline-none focus:ring-1 focus:ring-[var(--color-primary-500)]/30 transition-all"
          />
        </div>
      </div>

      {/* ── Optional telemetry ────────────────────────────────── */}
      <div className="glass-card p-6 space-y-5">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold uppercase tracking-wider text-slate-400">
            Optional Metadata
          </h3>
          <div className="flex items-center gap-1 rounded-full bg-blue-500/10 px-2 py-0.5 text-[10px] font-medium text-blue-400">
            <Info className="h-3 w-3" /> Optional
          </div>
        </div>

        {/* GPS/IMU */}
        <div>
          <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-slate-300">
            <MapPin className="h-4 w-4 text-slate-500" />
            GPS / IMU Telemetry
          </label>
          <div className="flex items-center gap-3">
            <label className="flex-1 cursor-pointer rounded-xl border border-white/[0.08] bg-[var(--color-surface-700)] px-4 py-2.5 text-sm transition-all hover:border-white/[0.15]">
              <input
                type="file"
                accept=".csv,.srt,.json,.txt,.log"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && setTelemetry(e.target.files[0])}
              />
              <span className="text-slate-400">
                {telemetry ? telemetry.name : 'Choose CSV, SRT, or JSON…'}
              </span>
            </label>
            {telemetry && (
              <button
                onClick={() => setTelemetry(null)}
                className="rounded-lg p-2 text-slate-500 hover:text-red-400"
              >
                <X className="h-4 w-4" />
              </button>
            )}
          </div>
        </div>

        {/* Camera intrinsics */}
        <div>
          <label className="mb-1.5 flex items-center gap-2 text-sm font-medium text-slate-300">
            <Camera className="h-4 w-4 text-slate-500" />
            Camera Intrinsics (JSON)
          </label>
          <textarea
            value={cameraIntrinsics}
            onChange={(e) => setCameraIntrinsics(e.target.value)}
            placeholder='{"fx": 1000, "fy": 1000, "cx": 640, "cy": 360}'
            rows={2}
            className="w-full rounded-xl border border-white/[0.08] bg-[var(--color-surface-700)] px-4 py-2.5 text-sm text-slate-200 placeholder:text-slate-600 focus:border-[var(--color-primary-500)] focus:outline-none focus:ring-1 focus:ring-[var(--color-primary-500)]/30 transition-all font-mono"
          />
        </div>
      </div>

      {/* ── Submit button ─────────────────────────────────────── */}
      <button
        onClick={handleSubmit}
        disabled={!video || uploading}
        className="w-full flex items-center justify-center gap-2 rounded-xl bg-gradient-to-r from-[var(--color-primary-500)] to-[var(--color-primary-600)] px-6 py-3.5 text-base font-semibold text-white shadow-lg transition-all hover:shadow-[var(--color-primary-500)]/20 hover:scale-[1.01] active:scale-[0.99] disabled:opacity-40 disabled:cursor-not-allowed disabled:hover:scale-100"
      >
        {uploading ? (
          <>
            <Loader2 className="h-5 w-5 animate-spin" />
            Uploading…
          </>
        ) : (
          <>
            <UploadIcon className="h-5 w-5" />
            Upload & Create Job
          </>
        )}
      </button>
    </div>
  );
}

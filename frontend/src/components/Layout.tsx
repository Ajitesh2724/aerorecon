import { Outlet, Link, useLocation } from 'react-router-dom';
import {
  Home,
  Upload,
  Box,
  Activity,
  Github,
} from 'lucide-react';

const NAV_ITEMS = [
  { to: '/', label: 'Dashboard', icon: Home },
  { to: '/upload', label: 'New Scan', icon: Upload },
];

export default function Layout() {
  const location = useLocation();

  return (
    <div className="flex min-h-screen">
      {/* ── Sidebar ───────────────────────────────────────────── */}
      <aside className="fixed left-0 top-0 z-40 flex h-screen w-64 flex-col border-r border-white/[0.06] bg-[var(--color-surface-800)]">
        {/* Brand */}
        <Link to="/" className="flex items-center gap-3 px-6 py-5">
          <div className="flex h-10 w-10 items-center justify-center rounded-xl bg-gradient-to-br from-[var(--color-primary-500)] to-[var(--color-accent-500)]">
            <Box className="h-5 w-5 text-white" />
          </div>
          <div>
            <h1 className="text-lg font-bold tracking-tight gradient-text">
              AeroRecon
            </h1>
            <p className="text-[10px] uppercase tracking-widest text-slate-500">
              3D Reconstruction
            </p>
          </div>
        </Link>

        {/* Navigation */}
        <nav className="mt-2 flex-1 space-y-1 px-3">
          {NAV_ITEMS.map(({ to, label, icon: Icon }) => {
            const active = location.pathname === to;
            return (
              <Link
                key={to}
                to={to}
                className={`flex items-center gap-3 rounded-xl px-4 py-2.5 text-sm font-medium transition-all duration-200 ${
                  active
                    ? 'bg-[var(--color-primary-500)]/10 text-[var(--color-primary-400)]'
                    : 'text-slate-400 hover:bg-white/[0.04] hover:text-slate-200'
                }`}
              >
                <Icon className="h-4 w-4" />
                {label}
              </Link>
            );
          })}
        </nav>

        {/* Footer */}
        <div className="border-t border-white/[0.06] px-4 py-4">
          <div className="flex items-center gap-2 text-xs text-slate-500">
            <Activity className="h-3 w-3" />
            <span>SIH 2026 — Problem 26158</span>
          </div>
          <a
            href="https://github.com/Ajitesh2724/aerorecon"
            target="_blank"
            rel="noopener noreferrer"
            className="mt-2 flex items-center gap-2 text-xs text-slate-500 hover:text-slate-300 transition-colors"
          >
            <Github className="h-3 w-3" />
            Ajitesh2724/aerorecon
          </a>
        </div>
      </aside>

      {/* ── Main content area ─────────────────────────────────── */}
      <main className="ml-64 flex-1 min-h-screen">
        <Outlet />
      </main>
    </div>
  );
}

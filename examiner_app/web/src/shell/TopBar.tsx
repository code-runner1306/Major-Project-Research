// Slim top bar, shown only below the narrow breakpoint. Its button opens the experiment drawer.
export function TopBar({ open, onToggle }: { open: boolean; onToggle: () => void }) {
  return (
    <header className="topbar">
      <button
        type="button"
        className="nav-toggle"
        aria-label={open ? 'Close experiment menu' : 'Open experiment menu'}
        aria-expanded={open}
        aria-controls="site-nav"
        onClick={onToggle}
      >
        <svg viewBox="0 0 24 24" width="22" height="22" aria-hidden="true">
          <path d="M3 6h18M3 12h18M3 18h18" stroke="currentColor" strokeWidth="2" fill="none" />
        </svg>
      </button>
      <span className="topbar-title">GDN Decay Granularity · Examiner</span>
    </header>
  );
}

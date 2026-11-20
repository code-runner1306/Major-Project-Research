import { useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { EXPERIMENTS } from '../experiments';
import type { ExperimentDef, ExperimentGroup } from '../types';
import { StatusChip } from '../components/StatusChip';
import { NARROW_QUERY, useMediaQuery } from './useMediaQuery';

const GROUP_LABELS: Record<ExperimentGroup, string> = {
  start: 'Start',
  experiments: 'Experiments',
  appendix: 'Appendix',
};

const GROUP_ORDER: ExperimentGroup[] = ['start', 'experiments', 'appendix'];

function Item({ def, active, onNavigate }: { def: ExperimentDef; active: boolean; onNavigate: () => void }) {
  return (
    <li>
      <a
        href={`#/${def.id}`}
        className={active ? 'nav-item active' : 'nav-item'}
        aria-current={active ? 'page' : undefined}
        onClick={onNavigate}
      >
        <span className="nav-title">{def.title}</span>
        <StatusChip status={def.status} />
      </a>
    </li>
  );
}

// Wide screens: a static sidebar. Below 900px it is a drawer: hidden until `open`, closed by the
// backdrop, Escape, or choosing an experiment.
export function Sidebar({
  activeId,
  open,
  onClose,
}: {
  activeId: string;
  open: boolean;
  onClose: () => void;
}) {
  const narrow = useMediaQuery(NARROW_QUERY);
  const reduced = useReducedMotion();
  const hidden = narrow && !open;
  // Appendix is collapsed by default.
  const [appendixOpen, setAppendixOpen] = useState(false);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open, onClose]);

  return (
    <>
      <AnimatePresence>
        {narrow && open && (
          <motion.div
            key="drawer-backdrop"
            className="drawer-backdrop"
            aria-hidden="true"
            onClick={onClose}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            transition={{ duration: reduced ? 0 : 0.2 }}
          />
        )}
      </AnimatePresence>
      <motion.nav
        id="site-nav"
        className="sidebar"
        aria-label="Experiments"
        inert={hidden}
        initial={false}
        animate={{ x: hidden ? '-100%' : '0%' }}
        transition={{ duration: reduced ? 0 : 0.28, ease: [0.22, 1, 0.36, 1] }}
      >
        <div className="sidebar-brand">
          <div className="brand-title">GDN Decay Granularity</div>
          <div className="brand-sub">Examiner</div>
        </div>

        {GROUP_ORDER.map((group) => {
          const items = EXPERIMENTS.filter((e) => e.group === group);
          if (group === 'appendix') {
            return (
              <div className="nav-group" key={group}>
                <button
                  type="button"
                  className="nav-group-toggle"
                  aria-expanded={appendixOpen}
                  aria-controls="nav-appendix"
                  onClick={() => setAppendixOpen((o) => !o)}
                >
                  <span>{GROUP_LABELS[group]}</span>
                  <span className="caret" aria-hidden="true">{appendixOpen ? '▾' : '▸'}</span>
                </button>
                {appendixOpen && (
                  <ul id="nav-appendix" className="nav-list">
                    {items.map((def) => (
                      <Item key={def.id} def={def} active={def.id === activeId} onNavigate={onClose} />
                    ))}
                  </ul>
                )}
              </div>
            );
          }
          return (
            <div className="nav-group" key={group}>
              <div className="nav-group-label">{GROUP_LABELS[group]}</div>
              <ul className="nav-list">
                {items.map((def) => (
                  <Item key={def.id} def={def} active={def.id === activeId} onNavigate={onClose} />
                ))}
              </ul>
            </div>
          );
        })}
      </motion.nav>
    </>
  );
}

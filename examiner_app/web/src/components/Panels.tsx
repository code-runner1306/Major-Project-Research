import { useId, useRef, useState, type KeyboardEvent, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import './story.css';

export interface PanelItem {
  label: string;
  content: ReactNode;
}

const SHIFT = 16;

// Small tabs above one visual at a time. Arrow keys, Home and End move between tabs (roving focus).
export function Panels({ items, label = 'Views' }: { items: PanelItem[]; label?: string }) {
  const reduce = useReducedMotion() ?? false;
  const baseId = useId();
  const [active, setActive] = useState(0);
  const dirRef = useRef(1);
  const tabRefs = useRef<(HTMLButtonElement | null)[]>([]);

  if (items.length === 0) return null;
  const current = Math.min(active, items.length - 1);

  const select = (i: number, focus: boolean) => {
    dirRef.current = i >= current ? 1 : -1;
    setActive(i);
    if (focus) tabRefs.current[i]?.focus();
  };

  const onKeyDown = (e: KeyboardEvent<HTMLButtonElement>) => {
    const n = items.length;
    let next: number | null = null;
    if (e.key === 'ArrowRight') next = (current + 1) % n;
    else if (e.key === 'ArrowLeft') next = (current - 1 + n) % n;
    else if (e.key === 'Home') next = 0;
    else if (e.key === 'End') next = n - 1;
    if (next === null) return;
    e.preventDefault();
    select(next, true);
  };

  const dir = dirRef.current;
  const panelId = `${baseId}-panel`;

  return (
    <div className="panels">
      <div role="tablist" aria-label={label} className="panels__tabs">
        {items.map((item, i) => (
          <button
            key={item.label}
            ref={(el) => {
              tabRefs.current[i] = el;
            }}
            id={`${baseId}-tab-${i}`}
            type="button"
            role="tab"
            className="panels__tab"
            aria-selected={i === current}
            aria-controls={panelId}
            tabIndex={i === current ? 0 : -1}
            onClick={() => select(i, false)}
            onKeyDown={onKeyDown}
          >
            {item.label}
          </button>
        ))}
      </div>
      <div className="panels__viewport">
        {reduce ? (
          <div key={current} id={panelId} role="tabpanel" aria-labelledby={`${baseId}-tab-${current}`} className="panels__panel">
            {items[current].content}
          </div>
        ) : (
          <AnimatePresence mode="wait" initial={false} custom={dir}>
            <motion.div
              key={current}
              id={panelId}
              role="tabpanel"
              aria-labelledby={`${baseId}-tab-${current}`}
              className="panels__panel"
              variants={{
                enter: (d: number) => ({ opacity: 0, x: d * SHIFT }),
                center: { opacity: 1, x: 0 },
                exit: (d: number) => ({ opacity: 0, x: d * -SHIFT }),
              }}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.2, ease: 'easeOut' }}
            >
              {items[current].content}
            </motion.div>
          </AnimatePresence>
        )}
      </div>
    </div>
  );
}

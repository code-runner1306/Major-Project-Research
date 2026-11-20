import { useRef } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { ARM_LABEL, type LayoutFixSummary } from '../../api';
import { WhatThisShows } from '../../components/WhatThisShows';
import { shortcutCounts, swatch } from './derive';

/** One dot per run (scalar seeds, then channel-wise seeds). A filled dot means that run shows the shortcut. */
export function ShortcutDots({ s }: { s: LayoutFixSummary }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  const reduce = useReducedMotion() ?? false;
  const rows = shortcutCounts(s);
  const perRow = Math.max(1, ...rows.map((r) => r.runs.length));

  return (
    <div className="lf-dots" ref={ref}>
      {rows.map((row, armIndex) => (
        <div className="lf-dots__row" key={row.arm}>
          <div className="lf-dots__label">{ARM_LABEL[row.arm]}</div>
          <div className="lf-dots__cells">
            {row.runs.map((run, i) => {
              // Sequence across the whole grid: each filled dot pops in after the one before it.
              const order = armIndex * perRow + i;
              const fill = swatch(row.arm, run.variant);
              return (
                <span
                  key={`${run.variant}-${run.seed}`}
                  className={`lf-dot${run.variant === 'channel-wise' ? ' lf-dot--channel' : ''}`}
                  title={`${ARM_LABEL[row.arm]}, ${run.variant}, seed ${run.seed}: ${run.shortcut ? 'shortcut' : 'no shortcut'}`}
                  aria-label={`${run.variant} seed ${run.seed}: ${run.shortcut ? 'has the shortcut' : 'no shortcut'}`}
                >
                  {run.shortcut && (
                    <motion.span
                      className="lf-dot__fill"
                      style={{ background: fill }}
                      initial={{ scale: 0 }}
                      animate={{ scale: inView ? 1 : 0 }}
                      transition={
                        reduce
                          ? { duration: 0 }
                          : { type: 'spring', stiffness: 320, damping: 18, delay: inView ? order * 0.09 : 0 }
                      }
                    />
                  )}
                </span>
              );
            })}
          </div>
          <div className="lf-dots__count">
            {row.shortcut} of {row.total}
          </div>
        </div>
      ))}
      <WhatThisShows>a filled dot is a run with the shortcut: block-first accuracy at least 20 points below general.</WhatThisShows>
    </div>
  );
}

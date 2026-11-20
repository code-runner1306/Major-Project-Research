import type { ReactNode } from 'react';
import { motion, useReducedMotion } from 'motion/react';

export interface ColumnBar {
  /** Short plain name shown under the column (wraps to two lines at most). */
  label: string;
  value: number;
  color: string;
}

export interface ColumnGroup {
  title: ReactNode;
  bars: ColumnBar[];
}

/**
 * Compact column chart: one group per experiment, one column per arm, side by side. Columns grow from zero on
 * first view. Used instead of stacked horizontal bars so every panel fits in the slide without scrolling.
 */
export function ColumnBars({ groups, format }: { groups: ColumnGroup[]; format: (v: number) => string }) {
  const reduce = useReducedMotion() ?? false;
  return (
    <div className="anat-cols">
      {groups.map((g, gi) => (
        <section key={gi} className="anat-cols__group">
          <h4>{g.title}</h4>
          <div className="anat-cols__plot" style={{ gridTemplateColumns: `repeat(${g.bars.length}, minmax(0, 1fr))` }}>
            {g.bars.map((b, i) => {
              const height = `${Math.max(0, Math.min(1, b.value)) * 100}%`;
              return (
                <div key={i} className="anat-cols__col">
                  <span className="anat-cols__value">{format(b.value)}</span>
                  <div className="anat-cols__slot">
                    <motion.div
                      className="anat-cols__bar"
                      style={{ background: b.color }}
                      initial={reduce ? false : { height: 0 }}
                      whileInView={{ height }}
                      viewport={{ once: true }}
                      transition={reduce ? { duration: 0 } : { duration: 0.8, delay: 0.12 * i, ease: 'easeOut' }}
                    />
                  </div>
                  <span className="anat-cols__name">{b.label}</span>
                </div>
              );
            })}
          </div>
        </section>
      ))}
    </div>
  );
}

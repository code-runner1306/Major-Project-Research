import { motion, useReducedMotion } from 'motion/react';
import { ColumnBars } from './ColumnBars';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import type { ArmSummary, CellId, CellSummary, Summary } from './types';

export const pct = (v: number | null, digits = 1) => (v === null ? '—' : `${(v * 100).toFixed(digits)}%`);

/** The three arms every cell compares, in a fixed order and colour. */
const ARM_STYLE: { match: (a: ArmSummary) => boolean; name: string; color: string }[] = [
  { match: (a) => a.id === 'ours_channel-wise', name: 'Channel-wise (ours)', color: 'var(--c-channel)' },
  { match: (a) => a.id === 'ours_scalar', name: 'Scalar (ours, added)', color: 'var(--c-scalar)' },
  { match: (a) => a.id === 'published_armed_gdn', name: 'Published channel-wise', color: 'var(--text-muted)' },
];

const CELL_ORDER: CellId[] = ['E1', 'E4', 'E5'];

function armsOf(cell: CellSummary) {
  return ARM_STYLE.map((s) => ({ style: s, arm: cell.arms.find(s.match) ?? null }));
}

/** Short names for the columns; the full arm names are in the MoreDetail text. */
const SHORT_NAME = ['Channel-wise (ours)', 'Scalar (ours)', 'Published channel-wise'];

/** Panel 1: mean accuracy per experiment, one column per arm. */
function MeanBars({ summary }: { summary: Summary }) {
  return (
    <div className="anat-meanbars">
      <ColumnBars
        format={(v) => pct(v)}
        groups={CELL_ORDER.flatMap((id) => {
          const cell = summary.cells.find((c) => c.id === id);
          if (!cell) return [];
          return [{
            title: cell.title,
            bars: armsOf(cell).map(({ style, arm }, i) => ({
              label: SHORT_NAME[i],
              value: arm?.mean ?? 0,
              color: style.color,
            })),
          }];
        })}
      />
      <WhatThisShows>average accuracy over seeds, one column per arm of each experiment.</WhatThisShows>
    </div>
  );
}

/** Panel 2: every seed as a dot. The dashed line is the lock-in threshold for that experiment. */
function SeedDots({ summary }: { summary: Summary }) {
  const reduce = useReducedMotion() ?? false;
  return (
    <div className="anat-dots">
      {CELL_ORDER.map((id) => {
        const cell = summary.cells.find((c) => c.id === id);
        if (!cell) return null;
        const pctLine = cell.threshold * 100;
        return (
          <section key={id} className="anat-dots__cell">
            <h4>
              {cell.title} <span className="anat-muted">locks in at {Math.round(cell.threshold * 100)}%</span>
            </h4>
            {armsOf(cell).map(({ style, arm }) =>
              arm ? (
                <div className="anat-dots__row" key={style.name}>
                  <span className="anat-dots__label">{style.name}</span>
                  <div className="anat-dots__track">
                    <span className="anat-dots__threshold" style={{ left: `${pctLine}%` }} aria-hidden="true" />
                    {arm.per_seed.map((s, i) => (
                      <motion.span
                        key={s.seed}
                        className="anat-dots__dot"
                        title={`seed ${s.seed}: ${pct(s.accuracy)}`}
                        style={{ left: `${s.accuracy * 100}%`, background: style.color }}
                        initial={reduce ? false : { opacity: 0, scale: 0.4 }}
                        animate={{ opacity: 1, scale: 1 }}
                        transition={reduce ? { duration: 0 } : { duration: 0.35, delay: 0.04 * i }}
                      />
                    ))}
                  </div>
                  <span className="anat-dots__count">
                    {arm.lock_in}/{arm.n} locked in
                  </span>
                </div>
              ) : null,
            )}
          </section>
        );
      })}
      <WhatThisShows>one dot per training run; dots to the right of the dashed line locked in.</WhatThisShows>
    </div>
  );
}

/** Panel 3: the saved tests. Plain names; the mean test is the permutation test, lock-ins use Fisher's exact test. */
function TestsTable({ summary }: { summary: Summary }) {
  return (
    <div className="anat-tests">
      <table>
        <thead>
          <tr>
            <th scope="col">Experiment</th>
            <th scope="col">Channel-wise mean</th>
            <th scope="col">Scalar mean</th>
            <th scope="col">Channel minus scalar</th>
            <th scope="col">Mean test (p)</th>
            <th scope="col">Locked in (channel / scalar)</th>
            <th scope="col">Lock-in test (p)</th>
          </tr>
        </thead>
        <tbody>
          {CELL_ORDER.map((id) => {
            const t = summary.tests[id];
            const cell = summary.cells.find((c) => c.id === id);
            if (!t || !cell) return null;
            return (
              <tr key={id}>
                <th scope="row">{cell.title}</th>
                <td>{pct(t.channel_mean)}</td>
                <td>{pct(t.scalar_mean)}</td>
                <td className={t.mean_difference > 0 ? 'anat-good' : 'anat-bad'}>
                  {(t.mean_difference * 100).toFixed(1)} pts
                </td>
                <td>{t.permutation_p.toFixed(4)}</td>
                <td>
                  {t.channel_lock_in} / {t.scalar_lock_in} of {t.n}
                </td>
                <td>{t.lock_in_fisher_p.toFixed(2)}</td>
              </tr>
            );
          })}
        </tbody>
      </table>
      <WhatThisShows>negative differences mean scalar decay scored higher; p below 0.05 means the gap is unlikely by chance.</WhatThisShows>
    </div>
  );
}

export function ResultsVisual({ summary }: { summary: Summary }) {
  return (
    <div className="anat-results">
      <Panels
        label="Results"
        items={[
          { label: 'Mean accuracy', content: <MeanBars summary={summary} /> },
          { label: 'Every seed', content: <SeedDots summary={summary} /> },
          { label: 'Tests', content: <TestsTable summary={summary} /> },
        ]}
      />
    </div>
  );
}

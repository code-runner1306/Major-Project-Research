import { useRef } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { ARM_LABEL, shortcutRuns, type Arm, type LayoutFixSummary } from '../../api';
import { AnimatedBars } from '../../components/AnimatedBars';
import { AnimatedLines, type LineSeries } from '../../components/AnimatedLines';
import { WhatThisShows } from '../../components/WhatThisShows';
import { answerShares, armColor, blockFirstBars, fmtPct, swatch } from './derive';

/** (a) Block-first, 3-rewrite accuracy for every arm and decay granularity. */
export function BlockFirstBars({ s }: { s: LayoutFixSummary }) {
  const data = blockFirstBars(s).map((b) => ({
    label: `${ARM_LABEL[b.arm]} · ${b.variant}`,
    value: b.value,
    color: swatch(b.arm, b.variant),
  }));
  return (
    <>
      <AnimatedBars data={data} max={1} format={fmtPct} />
      <WhatThisShows>fresh block-first examples, 3 rewrites. Solid bars are scalar decay; hatched bars are channel-wise.</WhatThisShows>
    </>
  );
}

/** (c) Validation accuracy per epoch for all 30 runs. */
export function CurvesPanel({ s }: { s: LayoutFixSummary }) {
  const series: LineSeries[] = s.curves.map((c) => ({
    name: `${ARM_LABEL[c.arm]} · ${c.variant} · seed ${c.seed}`,
    ys: c.dev_accuracy,
    color: armColor(c.arm),
    dashed: c.variant === 'channel-wise',
  }));
  const xs = s.curves[0]?.epochs;
  return (
    <>
      <AnimatedLines
        series={series}
        xs={xs}
        yMax={1}
        yLabel="validation accuracy"
        xLabel="epoch"
      />
      <WhatThisShows>every run ends near the top, so this ordinary test cannot tell the arms apart.</WhatThisShows>
    </>
  );
}

const SOURCE_COLOR: Record<string, string> = {
  latest: 'var(--c-good)',
  first_rewrite: 'color-mix(in srgb, var(--c-channel) 85%, var(--text))',
  second_rewrite: 'var(--c-channel)',
  third_rewrite: 'color-mix(in srgb, var(--c-channel) 55%, var(--surface))',
  initial: 'var(--border-strong)',
  other: 'var(--border)',
};
const SOURCE_LABEL: Record<string, string> = {
  latest: 'latest value (correct)',
  first_rewrite: 'first rewrite',
  second_rewrite: 'second rewrite',
  third_rewrite: 'third rewrite',
  initial: 'original value',
  other: 'other',
};

/** (d) Where the answers come from: stacked shares per arm for block-first, 3-rewrite fresh data. */
export function AnswerSources({ s }: { s: LayoutFixSummary }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  const reduce = useReducedMotion() ?? false;
  const rows = answerShares(s);
  const keys = Array.from(new Set(rows.flatMap((r) => r.shares.map((x) => x.key))));
  const ordered = [...keys].sort((a, b) => Object.keys(SOURCE_COLOR).indexOf(a) - Object.keys(SOURCE_COLOR).indexOf(b));

  return (
    <div className="lf-stack" ref={ref}>
      {rows.map((row) => (
        <div className="lf-stack__row" key={row.arm}>
          <div className="lf-stack__label">{ARM_LABEL[row.arm]}</div>
          <div className="lf-stack__bar" role="img" aria-label={`${ARM_LABEL[row.arm]} answer sources`}>
            {ordered.map((key, i) => {
              const share = row.shares.find((x) => x.key === key)?.share ?? 0;
              return (
                <motion.span
                  key={key}
                  className="lf-stack__seg"
                  title={`${SOURCE_LABEL[key] ?? key}: ${fmtPct(share)}`}
                  style={{ background: SOURCE_COLOR[key] ?? 'var(--border)' }}
                  initial={{ width: '0%' }}
                  animate={{ width: inView ? `${share * 100}%` : '0%' }}
                  transition={reduce ? { duration: 0 } : { duration: 0.9, ease: 'easeOut', delay: i * 0.08 }}
                />
              );
            })}
          </div>
        </div>
      ))}
      <div className="lf-stack__legend">
        {ordered.map((key) => (
          <span key={key}>
            <i style={{ background: SOURCE_COLOR[key] ?? 'var(--border)' }} />
            {SOURCE_LABEL[key] ?? key}
          </span>
        ))}
      </div>
      <WhatThisShows>which write each model's answer matched, on block-first examples with 3 rewrites.</WhatThisShows>
    </div>
  );
}

/** (e) Saved A4 report next to today's fresh rerun, for the 6 A4 runs on block-first, 3-rewrite data. */
export function RerunTable({ s }: { s: LayoutFixSummary }) {
  const arm: Arm = 'A4_distance_and_rewrites';
  const fresh = shortcutRuns(s, arm, 'fresh');
  const saved = shortcutRuns(s, arm, 'saved');
  const savedValue = (variant: string, seed: number) =>
    s.saved_a4_report.find(
      (r) => r.arm === arm && r.variant === variant && r.seed === seed && r.cell === 'main_r3' && r.layout === 'block_first',
    )?.latest;
  const freshValue = (variant: string, seed: number) =>
    s.fresh_results.find(
      (r) => r.arm === arm && r.variant === variant && r.seed === seed && r.cell === 'main_r3' && r.layout === 'block_first',
    )?.latest;
  const savedShortcuts = saved.filter((r) => r.shortcut).length;
  const freshShortcuts = fresh.filter((r) => r.shortcut).length;

  return (
    <>
      <table className="lf-table">
        <thead>
          <tr>
            <th>decay granularity</th>
            <th>seed</th>
            <th>saved report</th>
            <th>fresh rerun today</th>
          </tr>
        </thead>
        <tbody>
          {(['scalar', 'channel-wise'] as const).flatMap((variant) =>
            [42, 43, 44].map((seed) => (
              <tr key={`${variant}-${seed}`}>
                <td>{variant}</td>
                <td>{seed}</td>
                <td>{savedValue(variant, seed) !== undefined ? fmtPct(savedValue(variant, seed)!) : '–'}</td>
                <td>{freshValue(variant, seed) !== undefined ? fmtPct(freshValue(variant, seed)!) : '–'}</td>
              </tr>
            )),
          )}
        </tbody>
      </table>
      <p className="lf-muted">
        A4 runs showing the shortcut: saved report {savedShortcuts} of {saved.length}, fresh rerun {freshShortcuts} of {fresh.length}.
      </p>
      <WhatThisShows>block-first accuracy with 3 rewrites: the saved report against today's fresh rerun.</WhatThisShows>
    </>
  );
}

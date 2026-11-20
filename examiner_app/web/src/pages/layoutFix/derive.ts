// Small pure helpers for the layout-fix page. Every number shown on the page is derived here from the API response.
import {
  ARMS,
  ARM_LABEL,
  cellAccuracy,
  shortcutRuns,
  type Arm,
  type LayoutFixSummary,
  type TryLayout,
  type TryQuery,
  type Variant,
} from '../../api';

/** Arm colours, used consistently everywhere on the page. */
export const ARM_COLOR: Record<Arm, string> = {
  A0_control: 'var(--c-arm-a0)',
  A1_varied_distance: 'var(--c-arm-a1)',
  A2_no_block_first: 'var(--c-arm-a2)',
  A3_mixed_rewrites: 'var(--c-arm-a3)',
  A4_distance_and_rewrites: 'var(--c-arm-a4)',
};

export function armColor(arm: string): string {
  return ARM_COLOR[arm as Arm] ?? 'var(--c-arm-a0)';
}

/** Channel-wise runs are drawn hatched in the arm colour; scalar runs are solid. */
export function hatched(color: string): string {
  return `repeating-linear-gradient(-45deg, ${color} 0 3px, var(--surface) 3px 6px)`;
}

export function swatch(arm: string, variant: Variant | string): string {
  return variant === 'channel-wise' ? hatched(armColor(arm)) : armColor(arm);
}

export function fmtPct(x: number): string {
  return `${Number((x * 100).toFixed(1))}%`;
}

export function armName(arm: string): string {
  return ARM_LABEL[arm as Arm] ?? arm;
}

export const WHICH_WRITE_TEXT: Record<TryQuery['which_write'], string> = {
  latest: 'returned the latest value (correct)',
  first_rewrite: 'returned the first rewrite (wrong)',
  second_rewrite: 'returned the second rewrite (wrong)',
  third_rewrite: 'returned the third rewrite (wrong)',
  initial: 'returned the original value (wrong)',
  other: 'returned a value never written for this key (wrong)',
};

export function whichWriteText(w: TryQuery['which_write']): string {
  return WHICH_WRITE_TEXT[w] ?? `returned ${w}`;
}

/** Layouts that exist for a given number of rewrites (initial_then_run needs at least two rewrites). */
export const TRY_LAYOUTS: { id: TryLayout; label: string }[] = [
  { id: 'block_first', label: 'Block-first: every write of the queried key comes first' },
  { id: 'general_nonblock', label: 'General: writes interleaved, no block' },
  { id: 'block_after_one_unrelated', label: 'Block after one unrelated write' },
  { id: 'run_broken_before_latest', label: 'Run broken before the latest write' },
  { id: 'initial_then_run', label: 'Initial write, then a run (2+ rewrites)' },
];

export function layoutsFor(rewrites: number): TryLayout[] {
  return TRY_LAYOUTS.map((l) => l.id).filter((id) => id !== 'initial_then_run' || rewrites >= 2);
}

/** True when every T (queried-key write) comes before every U (other write). */
export function isBlockFirst(pattern: string): boolean {
  const lastT = pattern.lastIndexOf('T');
  const firstU = pattern.indexOf('U');
  return firstU === -1 || lastT < firstU;
}

/** Block-first 3-rewrite accuracy of every arm and decay granularity (fresh data). */
export function blockFirstBars(s: LayoutFixSummary) {
  return ARMS.flatMap((arm) =>
    (['scalar', 'channel-wise'] as Variant[]).map((variant) => ({
      arm,
      variant,
      value: cellAccuracy(s, arm, 'main_r3', 'block_first', variant),
    })),
  );
}

/** Number of runs per arm that show the shortcut, out of the total runs for that arm. */
export function shortcutCounts(s: LayoutFixSummary) {
  return ARMS.map((arm) => {
    const runs = shortcutRuns(s, arm);
    return { arm, shortcut: runs.filter((r) => r.shortcut).length, total: runs.length, runs };
  });
}

/** Answer-source shares for block-first, 3-rewrite fresh examples, per arm (summed over runs). */
export function answerShares(s: LayoutFixSummary) {
  return ARMS.map((arm) => {
    const totals: Record<string, number> = {};
    for (const r of s.fresh_results) {
      if (r.arm !== arm || r.cell !== 'main_r3' || r.layout !== 'block_first') continue;
      for (const [key, count] of Object.entries(r.answer_source)) totals[key] = (totals[key] ?? 0) + count;
    }
    const sum = Object.values(totals).reduce((a, b) => a + b, 0);
    const shares = Object.entries(totals).map(([key, count]) => ({ key, share: sum > 0 ? count / sum : 0 }));
    return { arm, sum, shares };
  });
}

export function finalAccuracyRange(s: LayoutFixSummary) {
  const finals = s.curves.map((c) => c.final_dev);
  return { lo: Math.min(...finals), hi: Math.max(...finals) };
}

/** Largest absolute value across a 3-D array of numbers (used to fix a shared heatmap scale). */
export function maxAbs3(xs: number[][][]): number {
  let m = 0;
  for (const plane of xs) for (const row of plane) for (const v of row) if (Number.isFinite(v)) m = Math.max(m, Math.abs(v));
  return m || 1;
}

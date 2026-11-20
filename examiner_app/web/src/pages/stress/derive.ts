// Labels and selectors over the stress summary. Every number shown on the page passes through here from the API.
import type { Variant } from '../../api';
import type { StressCell, StressCondition, StressFamily, StressSummary, StartLabel } from './types';

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** A fraction difference shown in percentage points, with a sign. */
export function pointsSigned(v: number, digits = 1): string {
  const x = v * 100;
  const sign = x > 0 ? '+' : x < 0 ? '−' : '';
  return `${sign}${Math.abs(x).toFixed(digits)} pts`;
}

export const START_LABEL: Record<StartLabel, string> = { default: 'Default start', 'mean-decay': 'Mean-decay start' };
export const VARIANT_LABEL: Record<Variant, string> = { scalar: 'Scalar decay', 'channel-wise': 'Channel-wise decay' };

/** The four trained settings, in the order the page lists them. */
export const SETTINGS: { init: StartLabel; variant: Variant }[] = [
  { init: 'default', variant: 'scalar' },
  { init: 'default', variant: 'channel-wise' },
  { init: 'mean-decay', variant: 'scalar' },
  { init: 'mean-decay', variant: 'channel-wise' },
];

/** Short form for narrow bar labels: 'Default · scalar', 'Mean-decay · channel-wise'. */
export function shortSettingLabel(init: string, variant: Variant): string {
  const start = init === 'mean-decay' ? 'Mean-decay' : 'Default';
  return `${start} · ${variant}`;
}

export function settingLabel(init: string, variant: Variant): string {
  return `${START_LABEL[init as StartLabel] ?? init} · ${VARIANT_LABEL[variant]}`;
}

/** Stress cells in the order the saved protocol lists them. */
export const OVERWRITE_IDS = ['baseline', 'more_keys', 'farther_query', 'more_rewrites'];
export const MQAR_IDS = ['baseline', 'more_keys', 'longer_context', 'longer_more_keys'];

export function cellOf(s: StressSummary, family: StressFamily, id: string): StressCell | undefined {
  return s.stress.cells.find((c) => c.family === family && c.id === id);
}

/** Plain description of a stress cell, built from its numbers. */
export function cellLabel(cell: StressCell | undefined): string {
  if (!cell) return '';
  if (cell.family === 'mqar') return `${cell.length}-token sequence, ${cell.keys} keys`;
  const rewrites = cell.rewrites === 1 ? '1 rewrite' : `${cell.rewrites} rewrites`;
  return `${cell.keys} keys, ${cell.distance} tokens to the query, ${rewrites}`;
}

/** One cell across the four trained settings, in SETTINGS order. */
export function settingsFor(s: StressSummary, family: StressFamily, cellId: string): StressCondition[] {
  return SETTINGS.map(({ init, variant }) => {
    const row = s.stress.conditions.find(
      (c) => c.family === family && c.cell === cellId && c.initialization === init && c.variant === variant,
    );
    if (!row) throw new Error(`Missing stress condition ${family}/${cellId}/${init}/${variant}`);
    return row;
  });
}

/** Mean over the four trained settings of one metric in one cell. */
export function averageOver(rows: StressCondition[], pick: (r: StressCondition) => number | null): number {
  const values = rows.map(pick).filter((v): v is number => v !== null);
  return values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
}

/** Gate-intervention means for one task and start, from the saved condition table. */
export function gateMeansFor(s: StressSummary, family: StressFamily, init: StartLabel) {
  return s.gate_interventions.condition_means.find((g) => g.family === family && g.initialization === init);
}

export function patchById(s: StressSummary, id: string) {
  return s.write_patches.patches.find((p) => p.id === id);
}

/** Plain label for a write-gate patch. Ids are never shown. */
export const PATCH_LABEL: Record<string, string> = {
  beta_one_latest: 'Full write strength on newest write',
  beta_zero_rewrite1: 'Block first replacement',
  key_align_latest: 'One address for every write',
  control_beta_one_unrelated: 'Control: full strength on an unrelated write',
};

// Response shapes for /api/shortcut/* (see ../../../../SPEC.md, "HTTP API"), plus the fixed labels the page uses.
// Accuracies and shares are fractions (0 to 1). Patch deltas are fractions too (0.162 means 16.2 points).
import type { Variant } from '../../api';

export type ShortcutLayout =
  | 'block_first'
  | 'general_nonblock'
  | 'block_after_one_unrelated'
  | 'run_broken_before_latest'
  | 'initial_then_run';

export type PatchId = 'none' | 'beta_one_latest' | 'key_align_latest' | 'beta_zero_rewrite1';

export interface Hypothesis {
  statement: string;
  /** Threshold as prose, written before the data was scored. */
  threshold: string;
  /** Threshold as a fraction in [0, 1], for the marker on the bar. */
  threshold_value: number;
  /** Raw counts from the saved results. The page never formats this object; it uses observed_text. */
  observed: unknown;
  /** One plain-English sentence computed on the server from observed. */
  observed_text: string;
  /** Observed result as a fraction in [0, 1], for the bar. */
  observed_value: number;
  confirmed: boolean;
}

export interface PerModel {
  init: string;
  variant: Variant;
  seed: number;
  local: boolean;
  block_first_r3: number | null;
  general_r3: number | null;
  block_first_r1: number | null;
  shortcut: boolean;
}

export interface PatchRow {
  patch: string;
  latest_delta: number;
  unrelated_delta: number;
  first_rewrite_delta: number;
  models_up: number;
  models_down: number;
}

export interface ShortcutSummary {
  hypotheses: Record<string, Hypothesis>;
  per_model: PerModel[];
  rule_agreement: ({ init: string; variant: Variant; seed: number } & Record<string, unknown>)[];
  patches: PatchRow[];
  stress_first_sign?: Record<string, unknown>;
  provenance?: unknown;
}

/** GET /api/shortcut/models: one row per local overwrite checkpoint. */
export interface ShortcutModel {
  init: string;
  variant: Variant;
  seed: number;
  checkpoint: string;
}

/** Which local model a control points at. */
export interface ModelRef {
  init: string;
  variant: Variant;
  seed: number;
}

export interface EvaluateRequest extends ModelRef {
  layout: ShortcutLayout;
  rewrites: number;
  distance: number;
  examples: number;
  example_seed: number;
  patch: PatchId;
  /** Examples to return (1 to 8, server default 3). */
  show_examples?: number;
}

/** One memory slot in write order. Slot numbers are 1-based. */
export interface ShortcutWrite {
  slot: number;
  key: number;
  value: number;
  /** A write of the queried key. */
  is_target: boolean;
  /** The newest write of the queried key. */
  is_latest_target: boolean;
}

export interface ShortcutExample {
  tokens: number[];
  roles: string[];
  prediction: number;
  gold?: number;
  which_write: string;
  rules_matched: string[];
  writes: ShortcutWrite[];
  question_key: number;
  gold_slot: number;
  /** 1-based slot whose value equals the prediction (most recent such slot); null if none. */
  picked_slot: number | null;
  tokens_between_last_write_and_question: number;
}

/** One row of pick_histogram. slot null counts answers that match no memory slot. */
export interface PickSlot {
  slot: number | null;
  count: number;
  is_target?: boolean;
  is_latest_target?: boolean;
}

/** A pick_histogram row that names a memory slot. */
export interface PickCard extends PickSlot {
  slot: number;
  is_target: boolean;
  is_latest_target: boolean;
}

export interface EvaluateResponse {
  accuracy_latest: number;
  accuracy_unrelated: number;
  first_rewrite_rate: number;
  answer_sources: Record<string, number>;
  rule_agreement: Record<string, number>;
  n_examples: number;
  slot_histogram: { slot: number; is_target: boolean; count: number; tokens_before_query: number }[];
  pick_histogram: PickSlot[];
  attribution: { layer: number; target_share_correct: number; target_share_wrong: number }[] | null;
  examples: ShortcutExample[];
}

export const SUMMARY_URL = '/api/shortcut/summary';
export const MODELS_URL = '/api/shortcut/models';
export const EVALUATE_URL = '/api/shortcut/evaluate';

export const DEFAULT_MODEL: ModelRef = { init: 'default', variant: 'scalar', seed: 42 };

export const LAYOUTS: { id: ShortcutLayout; label: string }[] = [
  { id: 'block_first', label: 'Block-first: every write of the queried key comes first' },
  { id: 'general_nonblock', label: 'General: writes interleaved, no block' },
  { id: 'block_after_one_unrelated', label: 'Block after one unrelated write' },
  { id: 'run_broken_before_latest', label: 'Run broken just before the latest write' },
  { id: 'initial_then_run', label: 'Initial write, then a run (2+ rewrites)' },
];

export function layoutsFor(rewrites: number): ShortcutLayout[] {
  return LAYOUTS.map((l) => l.id).filter((id) => id !== 'initial_then_run' || rewrites >= 2);
}

export const PATCHES: { id: PatchId; label: string; plain: string }[] = [
  { id: 'none', label: 'No patch', plain: 'The model exactly as it was trained.' },
  {
    id: 'beta_one_latest',
    label: 'Write strength 1 on the latest write',
    plain: 'Forces the write gate (β) to 1 for the newest write, so it overwrites fully.',
  },
  {
    id: 'key_align_latest',
    label: 'Earlier writes use the latest key',
    plain: "Makes every earlier write use exactly the newest write's key.",
  },
  {
    id: 'beta_zero_rewrite1',
    label: 'Switch off the first replacement',
    plain: 'Sets the write gate (β) to 0 for the first rewrite, so that write is never stored.',
  },
];

export const RULES: { id: string; name: string; plain: string }[] = [
  { id: 'latest', name: 'The latest write', plain: 'Answer the newest value written for the key. This is the correct rule.' },
  { id: 'initial', name: 'The first write', plain: 'Answer the original value, before any rewrite.' },
  { id: 'second_occurrence', name: 'The second write', plain: 'Answer the second time this key was written.' },
  { id: 'memory_slot_1', name: 'Memory slot 1', plain: 'Answer whatever sits in memory slot 1, counted from the start.' },
  {
    id: 'opening_run_second',
    name: 'Second item of the opening run',
    plain: "If the key's writes form one unbroken run from the start, answer the second write. Otherwise answer the latest.",
  },
  {
    id: 'final_run_second',
    name: 'Second item of the final run',
    plain: "Answer the second write in the key's last unbroken run of writes.",
  },
];

export const RULE_BY_ID: Record<string, { name: string; plain: string }> = Object.fromEntries(
  RULES.map((r) => [r.id, { name: r.name, plain: r.plain }]),
);

export const WHICH_WRITE_TEXT: Record<string, string> = {
  latest: 'returned the latest value (correct)',
  initial: 'returned the original value (wrong)',
  first_rewrite: 'returned the first replacement (wrong)',
  second_rewrite: 'returned the second rewrite (wrong)',
  third_rewrite: 'returned the third rewrite (wrong)',
  other: 'returned a value never written for this key (wrong)',
};

export function whichWriteText(which: string): string {
  return WHICH_WRITE_TEXT[which] ?? `returned ${which}`;
}

/** Display name of a training init. The folder label "calibrated" is shown as "mean-decay". */
export function initLabel(init: string): string {
  if (init === 'default') return 'default init';
  if (init === 'mean-decay' || init === 'calibrated') return 'mean-decay init';
  if (init === 'joint') return 'joint init';
  return init;
}

export function sameModel(a: ModelRef, b: ModelRef): boolean {
  return a.init === b.init && a.variant === b.variant && a.seed === b.seed;
}

export function modelKey(m: ModelRef): string {
  return `${m.init}|${m.variant}|${m.seed}`;
}

export function modelLabel(m: ModelRef): string {
  return `${initLabel(m.init)} · ${m.variant} · seed ${m.seed}`;
}

/** The default model if it is among the local checkpoints, otherwise the first local one. */
export function pickDefault(models: ModelRef[]): ModelRef | null {
  return models.find((m) => sameModel(m, DEFAULT_MODEL)) ?? models[0] ?? null;
}

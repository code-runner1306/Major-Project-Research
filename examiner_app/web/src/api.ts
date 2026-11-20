// Response types for the examiner server (see ../../SPEC.md "HTTP API"). Shared by all pages; edit only with SPEC.

export type Variant = 'scalar' | 'channel-wise';
export const ARMS = ['A0_control', 'A1_varied_distance', 'A2_no_block_first', 'A3_mixed_rewrites', 'A4_distance_and_rewrites'] as const;
export type Arm = (typeof ARMS)[number];
export const ARM_LABEL: Record<Arm, string> = {
  A0_control: 'A0 control', A1_varied_distance: 'A1 varied distance', A2_no_block_first: 'A2 no block-first',
  A3_mixed_rewrites: 'A3 mixed rewrites', A4_distance_and_rewrites: 'A4 distance + rewrites',
};
export const ARM_CHANGE: Record<Arm, string> = {
  A0_control: 'Original data: query always 8 tokens after the memory, always exactly 1 rewrite',
  A1_varied_distance: 'Query distance varies from 8 to 16 tokens',
  A2_no_block_first: 'Block-first layouts removed from the training data',
  A3_mixed_rewrites: 'Number of rewrites varies from 1 to 3',
  A4_distance_and_rewrites: 'Both changes: distance 8–16 and 1–3 rewrites',
};

export interface FreshResult { arm: Arm; variant: Variant; seed: number; cell: 'main_r1' | 'main_r3'; layout: 'block_first' | 'general_nonblock'; latest: number; answer_source: Record<string, number>; examples: number }
export interface DeltaTrace {
  layer: number; step: number; token: number; shapes: Record<string, number[]>; beta: number; alpha: number[]; k: number[];
  v_head: number[]; read_before_head: number[]; residual_norm: number; v_norm: number; read_error_before: number; read_error_after: number;
  state_before: number[][]; state_decayed: number[][]; state_after: number[][]; state_norm_before: number; state_norm_after: number;
  update_rank: number; max_abs_error_vs_model_recurrence: number;
}
export interface ShowcaseExample { tokens: number[]; roles: string[]; pattern: string; write_values: number[]; latest_label_position: number; rule_answers: Record<string, number>; predictions: Record<string, number> }
export interface Curve { arm: Arm; seed: number; variant: Variant; epochs: number[]; train_loss: number[]; dev_accuracy: number[]; arm_dev_accuracy: number[]; final_dev: number }
export interface GateLayer { layer: number; alpha_shape: number[]; a_proj_weight: number[]; A_log: number[]; dt_bias: number[]; alpha: number[][] }

/** GET /api/layout-fix/summary */
export interface LayoutFixSummary {
  meta: { fresh_seed_offset: number; examples_per_layout: number; generated: string; wall_seconds: number };
  fresh_results: FreshResult[];
  overlap_audit: Record<string, { fresh_examples: number; overlapping_by_arm: Record<Arm, number> } | null>;
  showcase: ShowcaseExample; showcase_trained_layout: ShowcaseExample;
  delta_rule: Record<string, DeltaTrace>; // keys like "A4_distance_and_rewrites/scalar"
  gates: Record<Variant, { parameters: number; layers: GateLayer[] }>;
  curves: Curve[];
  saved_a4_report: { arm: Arm; variant: Variant; seed: number; cell: string; layout: string; latest: number; examples: number }[];
  provenance?: { file: string; generated: string };
}

export type TryLayout = 'block_first' | 'general_nonblock' | 'block_after_one_unrelated' | 'run_broken_before_latest' | 'initial_then_run';
export interface TryRequest { arm: Arm; variant: Variant; seed: 42 | 43 | 44; layout: TryLayout; rewrites: number; distance: number; example_seed: number }
export interface TryQuery { position: number; kind: 'latest' | 'unrelated'; gold: number; prediction: number; which_write: 'latest' | 'initial' | 'first_rewrite' | 'second_rewrite' | 'third_rewrite' | 'other'; top5: { token: number; prob: number }[] }
/** POST /api/layout-fix/try */
export interface TryResponse {
  example: { tokens: number[]; roles: string[]; pattern: string; write_values: number[]; queries: TryQuery[] };
  internals: { layers: { layer: number; alpha: number[][]; beta: number[]; state_norm: number[] }[]; state_crops: number[][][]; write_positions: number[]; latest_write_position: number; recurrence_check_max_abs_diff: number };
  model: { arm: Arm; variant: Variant; seed: number; parameters: number; checkpoint: string };
}

const mean = (xs: number[]) => xs.reduce((a, b) => a + b, 0) / Math.max(xs.length, 1);

/** Mean latest-value accuracy of an arm on one fresh cell/layout (optionally one decay granularity). */
export function cellAccuracy(s: LayoutFixSummary, arm: Arm, cell: FreshResult['cell'], layout: FreshResult['layout'], variant?: Variant) {
  return mean(s.fresh_results.filter(r => r.arm === arm && r.cell === cell && r.layout === layout && (!variant || r.variant === variant)).map(r => r.latest));
}

/** Runs of an arm showing the shortcut: block-first 3-rewrite accuracy at least 20 points below general 3-rewrite accuracy. */
export function shortcutRuns(s: LayoutFixSummary, arm: Arm, source: 'fresh' | 'saved' = 'fresh') {
  const rows: { arm: string; variant: string; seed: number; cell: string; layout: string; latest: number }[] = source === 'fresh' ? s.fresh_results : s.saved_a4_report;
  const runs: { variant: Variant; seed: number; shortcut: boolean }[] = [];
  for (const variant of ['scalar', 'channel-wise'] as Variant[]) for (const seed of [42, 43, 44]) {
    const get = (layout: string) => rows.find(r => r.arm === arm && r.variant === variant && r.seed === seed && r.cell === 'main_r3' && r.layout === layout)?.latest;
    const bf = get('block_first'), gen = get('general_nonblock');
    if (bf !== undefined && gen !== undefined) runs.push({ variant, seed, shortcut: gen - bf >= 0.2 });
  }
  return runs;
}

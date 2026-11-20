// Response shapes for /api/length/* (server/length_data.py). Accuracies are fractions (0 to 1).
// Products are decay-only: alpha products omit the write strength, so they are not retention on their own.

export type LengthInit = 'default' | 'mean-decay';
export type LengthVariant = 'scalar' | 'channel-wise';
export type LengthSeed = 42 | 43 | 44;

export interface ModelRef {
  init: LengthInit;
  variant: LengthVariant;
  seed: LengthSeed;
}

export interface LocalModel extends ModelRef {
  checkpoint: string;
}

export interface SweepCell extends ModelRef {
  length: number;
  correct: number;
  total: number;
  accuracy: number;
  checkpoint_sha256: string;
  local: boolean;
}

export interface SweepModel extends ModelRef {
  local: boolean;
  /** Accuracy keyed by length as a string: "64", "256", "1024". */
  accuracy: Record<string, number>;
}

export interface LengthSummary {
  lengths: number[];
  training_length: number;
  prefix_tokens: number;
  examples_per_seed: number;
  cells: SweepCell[];
  models: SweepModel[];
  means: { length: number; mean: number; min: number; max: number; models: number }[];
  replays: (ModelRef & {
    length: number;
    correct: number;
    answers: number;
    accuracy: number;
    sweep_correct: number | null;
    matches_sweep: boolean | null;
    checkpoint_sha256: string;
  })[];
  replay_status: string;
  verification: {
    cells: number;
    verified_cells_agree: number;
    checkpoint_hashes: number | null;
    checkpoint_audit_status: string | null;
    local_checkpoints: number;
  };
  protocol: { description: string; lengths: number[]; examples_per_seed: number; device: string; torch: string };
  provenance: Record<string, string>;
}

export interface DecayLayer {
  layer: number;
  alpha_p50: number;
  half_life_tokens: number | null;
  product_p10: number;
  product_p50: number;
  product_p90: number;
  log10_product_p10: number;
  log10_product_p50: number;
  log10_product_p90: number;
  channels: number;
  examples: number;
}

export interface DecayRow extends ModelRef {
  length: number;
  filler_tokens: number;
  correct: number;
  total: number;
  layers: DecayLayer[];
}

export interface DecaySummary {
  rows: DecayRow[];
  histogram: {
    length: number;
    layer: number;
    examples_per_seed: number;
    initializer: string;
    inserted_tokens: number;
    records: { seed: number; correct: number; total: number; bin_edges: number[]; counts: number[] }[];
  };
  protocol: {
    seeds: number[];
    lengths: number[];
    examples_per_seed: number;
    prefix_tokens_before_filler: number;
    filler_token: number;
    alpha_definition: string;
    interpretation_guardrail: string;
  };
  provenance: Record<string, string>;
}

export interface RunRequest extends ModelRef {
  filler: number;
  examples: number;
}

export interface RunResponse {
  model: ModelRef & { checkpoint: string; parameters: number };
  filler: number;
  length: number;
  examples: number;
  data_seed: number;
  accuracy: number;
  correct: number;
  total: number;
  audit_passed: boolean;
  example: {
    tokens: number[];
    queries: { position: number; gold: number; prediction: number; correct: boolean }[];
    correct: number;
    total: number;
  };
  survival: {
    steps: number[];
    layers: { layer: number; log10_p10: number[]; log10_p50: number[]; log10_p90: number[] }[];
  };
  at_filler: { layer: number; product_p10: number; product_p50: number; product_p90: number; log10_p50: number; channels: number }[];
  saved: { correct: number; total: number; layers: { layer: number; log10_product_p50: number }[] } | null;
}

export const SUMMARY_URL = '/api/length/summary';
export const DECAY_URL = '/api/length/decay';
export const MODELS_URL = '/api/length/models';
export const RUN_URL = '/api/length/run';

/** Page labels for the saved folder names: "calibrated" on disk is "mean-decay" on the page. */
export const INIT_LABEL: Record<LengthInit, string> = { default: 'default', 'mean-decay': 'mean-decay' };

export const modelKey = (m: ModelRef) => `${m.init}|${m.variant}|${m.seed}`;
export const sameModel = (a: ModelRef, b: ModelRef) => modelKey(a) === modelKey(b);
export const modelLabel = (m: ModelRef) => `${INIT_LABEL[m.init]} · ${m.variant} · seed ${m.seed}`;

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** Longest filler the live run accepts (1,024 total tokens minus the 64-token training length). */
export const MAX_FILLER = 960;

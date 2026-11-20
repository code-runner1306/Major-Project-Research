// Response shapes for /api/decay-init/* (see server/routers/decay_init.py). Accuracies and decay values are fractions.

export type DecayInit = 'default' | 'mean-decay' | 'joint';
export type Granularity = 'scalar' | 'channel-wise';
export type Family = 'overwrite' | 'mqar';

export interface ModelRef {
  init: DecayInit;
  variant: Granularity;
  seed: number;
}

export interface MeasureRow {
  init: DecayInit;
  variant: Granularity;
  seeds: number[];
  mean: number;
  sd: number;
}

export interface Block {
  seeds: number[];
  mean: number;
  sd: number;
}

export interface SummaryFamily {
  measures: Record<'learning' | 'fresh' | 'fresh_retention', MeasureRow[] | undefined>;
  contrasts: Record<
    'learning' | 'fresh' | 'fresh_retention',
    { default: Block; mean_decay: Block; joint?: Block; interaction?: Block } | undefined
  >;
}

export interface Summary {
  seeds: number[];
  families: Record<Family, SummaryFamily>;
  definitions: Record<string, string>;
  provenance: { files: string[]; note: string };
}

export interface GateLayer {
  layer: number;
  trained: { mean: number; per_channel: number[]; channel_spread: number };
  start: null | {
    half_life: number;
    mean: number;
    per_channel: number[];
    channel_spread: number;
    maximum_error: number;
  };
}

export interface GateModel extends ModelRef {
  folder: string;
  checkpoint: string;
  input_sha256: string;
  layers: GateLayer[];
}

export interface Gates {
  generated: string;
  rows: number;
  definition: string;
  parameters: Record<Granularity, number>;
  models: GateModel[];
  provenance: { file: string };
}

export interface LocalModel {
  init: DecayInit;
  variant: Granularity;
  seed: number;
  checkpoint: string;
}

export interface TryRequest extends ModelRef {
  layout: string;
  rewrites: number;
  distance: number;
  example_seed: number;
}

export interface TryQuery {
  position: number;
  kind: 'latest' | 'unrelated';
  gold: number;
  prediction: number;
  which_write: string;
  top5: { token: number; prob: number }[];
}

export interface TryResponse {
  example: {
    tokens: number[];
    roles: string[];
    pattern: string;
    write_values: number[];
    queries: TryQuery[];
  };
  alpha: { layers: { layer: number; alpha: number[][] }[] };
  model: ModelRef & { parameters: number; checkpoint: string };
  start_mean: number[] | null;
}

export const SUMMARY_URL = '/api/decay-init/summary';
export const GATES_URL = '/api/decay-init/gates';
export const MODELS_URL = '/api/decay-init/models';
export const TRY_URL = '/api/decay-init/try';

export const LAYOUTS: { id: string; label: string }[] = [
  { id: 'general_nonblock', label: 'Mixed layout' },
  { id: 'block_first', label: 'Block-first layout' },
  { id: 'block_after_one_unrelated', label: 'One unrelated write first' },
  { id: 'run_broken_before_latest', label: 'Run broken before the newest write' },
  { id: 'initial_then_run', label: 'Original value, then a run' },
];

/** Layouts that exist for a given number of rewrites (mirrors the server rule). */
export function layoutsFor(rewrites: number): string[] {
  return rewrites >= 2 ? LAYOUTS.map((l) => l.id) : LAYOUTS.map((l) => l.id).filter((id) => id !== 'initial_then_run');
}

export const WHICH_WRITE_TEXT: Record<string, string> = {
  latest: 'the newest value',
  initial: 'the original value',
  first_rewrite: 'the first replacement',
  second_rewrite: 'the second replacement',
  third_rewrite: 'the third replacement',
  other: 'none of the values written',
};

export function initLabel(init: DecayInit): string {
  return init === 'mean-decay' ? 'mean-decay start' : 'default start';
}

export function granularityLabel(v: Granularity): string {
  return v === 'scalar' ? 'scalar decay' : 'channel-wise decay';
}

export function modelKey(m: ModelRef): string {
  return `${m.init}/${m.variant}/${m.seed}`;
}

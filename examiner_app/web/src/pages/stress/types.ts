// Response shapes for /api/stress/* (server/routers/stress.py, server/stress_data.py).
// Accuracies and rates are fractions (0 to 1). Write-patch changes are fractions too (0.1621 means 16.2 points).
import type { Variant } from '../../api';

export type StressFamily = 'mqar' | 'overwrite';
export type StartLabel = 'default' | 'mean-decay';

export interface StressCell {
  family: StressFamily;
  id: string;
  length?: number;
  keys: number;
  distance?: number;
  rewrites?: number;
}

export interface StressCondition {
  family: StressFamily;
  initialization: string;
  variant: Variant;
  cell: string;
  accuracy: number;
  accuracy_sd: number;
  retention: number | null;
  latest: number | null;
  stale: number | null;
}

export interface StressPerSeed {
  family: StressFamily;
  seed: number;
  initialization: string;
  variant: Variant;
  cell: string;
  accuracy: number;
  latest: number | null;
  retention: number | null;
  stale: number | null;
}

export interface GatePerModel {
  family: StressFamily;
  seed: number;
  initialization: string;
  baseline: number;
  shared_mean: number;
  static: number;
  shuffle_mean: number;
  shuffle_min: number;
  shuffle_max: number;
  layer_drops: (number | null)[];
}

export interface GateMeans {
  family: StressFamily;
  initialization: string;
  baseline: number;
  shared_mean: number;
  static: number;
  shuffle_mean: number;
}

export interface PatchModel {
  seed: number;
  initialization: string;
  variant: Variant;
  latest: number;
  retention: number;
}

export interface WritePatch {
  id: string;
  models: number;
  local_models: number;
  latest_change: number;
  retention_change: number;
  first_replacement_change: number;
  improved: number;
  worse: number;
  per_model: PatchModel[];
}

export interface StressSummary {
  stress: {
    cells: StressCell[];
    conditions: StressCondition[];
    per_seed: StressPerSeed[];
    query_distances: { seed: number; cell: string; mean: number; median: number; maximum: number }[];
    examples_per_condition: number;
  };
  gate_interventions: {
    shared_mean_per_model: { family: StressFamily; seed: number; initialization: string; baseline: number; shared_mean: number }[];
    per_model: GatePerModel[];
    condition_means: GateMeans[];
  };
  write_patches: {
    threshold_points: number;
    baseline: { models: number; latest: number; retention: number };
    patches: WritePatch[];
  };
  provenance: { files: string[]; generated: null };
}

/** GET /api/stress/models: one row per local overwrite checkpoint. */
export interface StressModel {
  init: string;
  variant: Variant;
  seed: number;
  checkpoint: string;
}

export type TryIntervention = 'none' | 'beta_one_latest' | 'beta_zero_rewrite1' | 'key_align_latest' | 'alpha_shared_mean';

/** GET /api/stress/interventions */
export interface Intervention {
  id: TryIntervention;
  label: string;
  detail: string;
}

export interface TryRequest {
  init: StartLabel;
  variant: Variant;
  seed: 42 | 43 | 44;
  intervention: TryIntervention;
  rewrites: number;
  distance: number;
  examples: number;
  example_seed: number;
  show: number;
}

export type WhichWrite = 'latest' | 'initial' | 'first_rewrite' | 'second_rewrite' | 'third_rewrite' | 'other';

export interface TryRates {
  latest_accuracy: number;
  unrelated_retention: number;
  first_replacement_rate: number;
  stale_rate: number;
}

export interface TryWrite {
  slot: number;
  key: number;
  value: number;
  is_target: boolean;
}

export interface TryQuery {
  kind: 'latest' | 'unrelated';
  gold: number;
  before: { prediction: number; which?: WhichWrite; picked_slot?: number | null; ok?: boolean };
  after: { prediction: number; which?: WhichWrite; picked_slot?: number | null; ok?: boolean };
}

export interface TryGateLayer {
  layer: number;
  alpha_mean: number[];
  alpha_min: number[];
  alpha_max: number[];
  beta: number[];
  state_norm: number[];
}

/** POST /api/stress/try */
export interface TryResponse {
  model: { init: string; variant: Variant; seed: number; parameters: number; checkpoint: string };
  setup: {
    intervention: TryIntervention;
    rewrites: number;
    distance: number;
    examples: number;
    example_seed: number;
    show: number;
    seed_offset: number;
    no_effect_expected: boolean;
  };
  before: TryRates;
  after: TryRates;
  changed_answers: number;
  fixed: number;
  broken: number;
  example: { tokens: number[]; roles: string[]; writes: TryWrite[]; queries: TryQuery[]; values: number[] };
  gates: { before: TryGateLayer[]; after: TryGateLayer[] };
}

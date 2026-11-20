// Shapes of the /api/anatomy responses (see examiner_app/server/anatomy_data.py and routers/anatomy.py).

export const SUMMARY_URL = '/api/anatomy/summary';
export const RUNS_URL = '/api/anatomy/runs';
export const EXAMPLE_URL = '/api/anatomy/example';

export type CellId = 'E1' | 'E4' | 'E5';
export type Decay = 'channel-wise' | 'scalar' | 'none';

export interface ArmSummary {
  id: string;
  label: string;
  decay: Decay;
  source: 'ours' | 'published';
  n: number;
  mean: number | null;
  std: number | null;
  lock_in: number;
  per_seed: { seed: number; accuracy: number }[];
  rebind_mean: number | null;
}

export interface CellSummary {
  id: CellId;
  title: string;
  collision: boolean;
  threshold: number;
  value_floor: number | null;
  arms: ArmSummary[];
}

export interface CellTest {
  channel_mean: number;
  scalar_mean: number;
  mean_difference: number;
  permutation_p: number;
  channel_lock_in: number;
  scalar_lock_in: number;
  lock_in_fisher_p: number;
  n: number;
}

export interface InterventionCell {
  n: number;
  means: Record<string, number>;
  runs: { seed: number; accuracy: Record<string, number>; file: string }[];
}

export interface Summary {
  provenance: { upstream: string; files: string[]; our_commits: string[]; frozen: string };
  predictions: Record<string, string>;
  cells: CellSummary[];
  tests: Record<CellId, CellTest>;
  channel_ahead_cells: CellId[];
  interventions: Partial<Record<CellId, InterventionCell>>;
  intervention_labels: Record<string, string>;
  channel_decay: Partial<Record<CellId, { values: number[]; file: string }>>;
  checkpoints_available: boolean;
}

export interface RunRow {
  cell: CellId;
  source: 'ours' | 'published';
  arm: string;
  label: string;
  decay: Decay;
  seed: number;
  eval_accuracy: number;
  rebind_accuracy: number | null;
  lock_in: boolean;
  value_floor: number | null;
  commit: string | null;
  file: string;
}

export interface Answer {
  position: number;
  answer: number;
  kind: string;
}

export interface AnatomyExample {
  generator: string;
  experiment: 'E4' | 'E5';
  tokens: number[];
  roles: string[];
  scored: Answer[];
  notes: string[];
  meta: Record<string, unknown>;
  reserved_tokens: Record<string, string>;
}

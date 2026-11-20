// Fetch helpers and types for the selective reset page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';

export const SUMMARY_URL = '/api/selective-reset/summary';
export const PROFILE_URL = '/api/selective-reset/profile';
export const TRY_URL = '/api/selective-reset/try';

export const QUERIES = ['target', 'survivor', 'rewrite', 'absent'] as const;
export type Query = (typeof QUERIES)[number];

/** Accuracy metrics for one memory load (fractions 0 to 1), as the pilot saved them. */
export interface LoadRow {
  load: number;
  reset_target: number;
  noop_target: number;
  reset_survivor: number;
  noop_survivor: number;
  reset_rewrite: number;
  noop_rewrite: number;
  reset_absent: number;
  noop_absent: number;
  reset_joint: number;
  noop_joint: number;
  stale_value_returned: number;
}

export interface RunSummary {
  family: string;
  decay: 'channel-wise' | 'scalar';
  seed: number;
  parameters: number;
  training_seconds: number;
  steps: number;
  batch: number;
  by_load: LoadRow[];
  has_checkpoint: boolean;
}

export interface Summary {
  task: {
    memory_loads: number[];
    keys_per_workspace: number;
    queries: string[];
    query_names: Record<string, string>;
    values: number;
    guess_rate: number;
    length: number;
    steps: number;
    batch: number;
  };
  runs: RunSummary[];
  per_query: Record<string, number>;
  by_load: LoadRow[];
  max_joint: number;
  max_survivor: number;
  min_reset_target: number;
  max_noop_target: number;
  live: boolean;
  checkpoints: { decay: string; seed: number; file: string }[];
  provenance: { results: string[]; task: string; checkpoint: string[] };
}

/** Share of answers per kind (fractions), keyed by the kinds listed in Profile.kinds. */
export type ProfileCell = Record<string, number>;

export interface Profile {
  episodes_per_cell: number;
  kinds: string[];
  checkpoint: { decay: string; seed: number };
  by_load: Record<string, Record<string, ProfileCell>>;
}

export interface TryToken {
  id: number;
  role: string;
  label: string;
}

export interface TryResult {
  load: number;
  reset: boolean;
  query: Query;
  seed: number;
  tokens: TryToken[];
  expected: number;
  expected_label: string;
  prediction: number;
  prediction_label: string;
  correct: boolean;
  top3: { token: number; label: string; prob: number }[];
  cleared_key_old_value: number;
  survivor_value: number;
  model: { decay: string; seed: number; parameters: number };
}

async function readDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
    if (body.detail !== undefined) return JSON.stringify(body.detail);
  } catch {
    // Non-JSON error body: fall back to the status line.
  }
  return `${res.status} ${res.statusText}`.trim();
}

/** One request. A network failure becomes an ApiFailure with status null, which useRetrying treats as "server starting". */
async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiFailure('The server did not answer.', null);
  }
  if (!res.ok) throw new ApiFailure(await readDetail(res), res.status);
  return (await res.json()) as T;
}

/** Saved results and whether the one checkpoint is present. Retries while the server starts. */
export function useSummary() {
  return useRetrying<Summary>('selective-reset-summary', () => request<Summary>(SUMMARY_URL));
}

/** The checkpoint's answers over fixed episodes. Only requested when the checkpoint is present (key null otherwise). */
export function useProfile(live: boolean) {
  return useRetrying<Profile>(live ? 'selective-reset-profile' : null, () => request<Profile>(PROFILE_URL));
}

export function runTry(body: { load: number; reset: boolean; query: Query; seed: number }): Promise<TryResult> {
  return request<TryResult>(TRY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

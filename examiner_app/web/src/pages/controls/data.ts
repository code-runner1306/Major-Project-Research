// Types and fetch hooks for the negative-controls page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';

export const SUMMARY_URL = '/api/controls/summary';
export const INSIDE_URL = '/api/controls/inside';
export const ZOOLOGY_EVALUATE_URL = '/api/controls/zoology/evaluate';

export type Variant = 'scalar' | 'channel-wise';
export type ZoologyInit = 'default' | 'mean-decay' | 'published';

export interface ZoologyModelRow {
  init: ZoologyInit;
  variant: Variant;
  seed: number;
  group: 'main' | 'published';
  local: boolean;
  checkpoint: string;
  /** Same rule as the saved test: a shifted layout at least 20 points below the standard one. */
  shortcut: boolean;
  accuracy: Record<string, number>;
}

export interface ZoologyLayout {
  id: string;
  label: string;
  description: string;
}

export interface ZoologyDesign {
  init: ZoologyInit;
  variant: Variant;
  models: number;
  accuracy: Record<string, number>;
}

export interface ZoologyAttribution {
  errors: number;
  other_memory_value: number;
  position_anchored: number;
}

export interface ZoologySummary {
  examples_per_cell: number;
  shortcut_gap: number;
  layouts: ZoologyLayout[];
  readings: {
    standard: string;
    shortcut: string;
    anchored: string;
    query_layout: Record<string, string>;
  };
  models: ZoologyModelRow[];
  models_total: number;
  models_local: number;
  shortcut_count: number;
  designs: ZoologyDesign[];
  attribution: Record<string, ZoologyAttribution>;
  hypotheses: Record<string, string>;
  verdicts: Record<string, unknown>;
  provenance: { files: Record<string, string> };
}

export interface KataSummary {
  learning_rates: string[];
  selected_learning_rate: string;
  parameters: { ours: number; paper: number | null } | null;
  replication: {
    ours: Record<string, number>;
    paper: Record<string, number>;
    ours_average: number;
    paper_average: number;
    adequate: boolean;
  };
  cells: Record<string, Record<string, number | null>>;
  cell_names: string[];
  verdicts: Record<string, unknown>;
  readings: { replication: string; shortcut: string; error_identity: string };
  shortcut_runs: number;
  runs_total: number;
  local_checkpoints: number;
  provenance: { files: Record<string, string> };
}

export interface ControlsSummary {
  zoology: ZoologySummary;
  kata: KataSummary;
}

export interface SlotCount {
  slot: number | null;
  count: number;
  from_queried_key?: number;
  from_other_key?: number;
}

export interface InsideZoology {
  init: ZoologyInit;
  variant: Variant;
  seed: number;
  group: 'main' | 'published';
  layout: string;
  n_answers: number;
  accuracy: number;
  saved_accuracy: number;
  matches_saved: boolean;
  answer_sources: Record<string, number>;
  errors: Record<string, number>;
  slot_histogram: SlotCount[];
}

export interface InsideShortcut {
  init: 'default' | 'mean-decay';
  variant: Variant;
  seed: number;
  layout: string;
  rewrites: number;
  n_answers: number;
  accuracy_latest: number;
  first_rewrite_rate: number | null;
  answer_sources: Record<string, number>;
  pick_histogram: { slot: number | null; count: number; is_target: boolean; is_latest_target: boolean }[];
  slot_histogram: SlotCount[];
}

export interface InsideFile {
  generated: string;
  zoology_settings: { examples: number; example_seed: string; layouts: string[] };
  shortcut_settings: { layout: string; rewrites: number; distance: number; examples: number; example_seed: number; patch: string };
  zoology: InsideZoology[];
  shortcut: InsideShortcut[];
  all_zoology_match_saved: boolean;
}

export interface TryQuery {
  position: number;
  gold: number;
  prediction: number;
  outcome: 'correct' | 'position_anchored' | 'other_pair' | 'other';
  picked_slot: number | null;
  gold_slot: number | null;
}

export interface TryExample {
  tokens: number[];
  roles: string[];
  queries: TryQuery[];
}

export interface ZoologyEvaluateRequest {
  init: ZoologyInit;
  variant: Variant;
  seed: number;
  layout: string;
  examples: number;
  example_seed: number;
  show_examples: number;
}

export interface ZoologyEvaluateResponse {
  model: { init: ZoologyInit; variant: Variant; seed: number; checkpoint: string };
  layout: string;
  layout_label: string;
  layout_description: string;
  examples: number;
  example_seed: number;
  n_answers: number;
  cell_seed: number;
  accuracy: number;
  accuracy_by_query: (number | null)[];
  answer_sources: Record<string, number>;
  errors: Record<string, number>;
  slot_histogram: SlotCount[];
  shown: TryExample[];
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

export function useControlsSummary() {
  return useRetrying<ControlsSummary>(SUMMARY_URL, () => request<ControlsSummary>(SUMMARY_URL));
}

export function useInside() {
  return useRetrying<InsideFile>(INSIDE_URL, () => request<InsideFile>(INSIDE_URL));
}

/** Live Zoology evaluation. Pass null to skip. The last settled request wins. */
export function useZoologyTry(body: ZoologyEvaluateRequest | null, delay = 220) {
  const key = body === null ? null : JSON.stringify(body);
  return useRetrying<ZoologyEvaluateResponse>(key, () =>
    request<ZoologyEvaluateResponse>(ZOOLOGY_EVALUATE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: String(key),
    }),
  delay);
}

/** Display label for a Zoology design, in the page's words. */
export function designLabel(init: ZoologyInit, variant: Variant): string {
  const initText = init === 'published' ? 'published' : init === 'mean-decay' ? 'mean-decay' : 'default';
  return `${initText}, ${variant}`;
}

export function modelLabel(m: { init: ZoologyInit; variant: Variant; seed: number }): string {
  return `${designLabel(m.init, m.variant)}, seed ${m.seed}`;
}

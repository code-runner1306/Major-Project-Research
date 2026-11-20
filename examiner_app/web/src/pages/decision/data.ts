// Fetch helpers and types for the decision fine-tuning page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';

export const SUMMARY_URL = '/api/decision/summary';

export interface DecisionVariant {
  variant: 'scalar' | 'channel-wise';
  first_train_loss: number;
  last_train_loss: number;
  mean_validation_soft_loss: number;
  steady_step_median_seconds: number;
  peak_allocated_gib: number | null;
  peak_reserved_gib: number | null;
  timed_steps: number;
}

export interface DecisionRun {
  id: 'short' | 'longer';
  file: string;
  steps: number;
  batch_size: number;
  max_length: number;
  train_examples: number;
  validation_examples: number;
  state_truncations: { train: number; validation: number };
  variants: DecisionVariant[];
}

export interface DecisionSummary {
  question: { workflow: string; question_id: string; labels: string[] };
  dataset: {
    split: string;
    fingerprint: string;
    available_train_cases: number;
    available_validation_cases: number;
  };
  device: string;
  torch: string;
  runs: DecisionRun[];
  pick_comparison_recorded: boolean;
  pre_training_validation_recorded: boolean;
  fine_tuned_weights_saved: boolean;
  base_checkpoints_local: string[];
  provenance: { files: string[]; script: string; note: string };
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
async function request<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiFailure('The server did not answer.', null);
  }
  if (!res.ok) throw new ApiFailure(await readDetail(res), res.status);
  return (await res.json()) as T;
}

/** Saved feasibility reports: labels, per-variant losses, step times and memory. */
export function useDecisionSummary() {
  return useRetrying<DecisionSummary>(SUMMARY_URL, () => request<DecisionSummary>(SUMMARY_URL));
}

/** Choice labels as plain words: "answer_directly" becomes "Answer directly". */
export function plainLabel(label: string): string {
  const words = label.replace(/_/g, ' ');
  return words.charAt(0).toUpperCase() + words.slice(1);
}

export function variantName(v: DecisionVariant['variant']): string {
  return v === 'channel-wise' ? 'channel-wise decay' : 'scalar decay';
}

export function runName(run: DecisionRun): string {
  return run.id === 'short' ? `Short run (${run.steps} steps)` : `Longer run (${run.steps} steps)`;
}

export const fixed = (v: number, digits = 3) => v.toFixed(digits);

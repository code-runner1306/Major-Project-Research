// Fetch helpers and hooks for the longer-sequences page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import { DECAY_URL, MODELS_URL, RUN_URL, SUMMARY_URL, type DecaySummary, type LengthSummary, type LocalModel, type RunRequest, type RunResponse } from './types';

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

/** Saved length sweep: 36 cells, replays, checkpoint audit. */
export function useLengthSummary() {
  return useRetrying<LengthSummary>(SUMMARY_URL, () => request<LengthSummary>(SUMMARY_URL));
}

/** Saved filler decay diagnostic: per layer and granularity, plus the product histogram. */
export function useDecaySummary() {
  return useRetrying<DecaySummary>(DECAY_URL, () => request<DecaySummary>(DECAY_URL));
}

/** The local MQAR checkpoints this machine can run. */
export function useLocalLengthModels() {
  return useRetrying<LocalModel[]>(MODELS_URL, () => request<LocalModel[]>(MODELS_URL));
}

/** Live run. Keyed on the request, so each change restarts the run; pass null to skip. */
export function useLengthRun(body: RunRequest | null, delay = 250) {
  const key = body === null ? null : JSON.stringify(body);
  return useRetrying<RunResponse>(key, () => request<RunResponse>(RUN_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: String(key),
  }), delay);
}

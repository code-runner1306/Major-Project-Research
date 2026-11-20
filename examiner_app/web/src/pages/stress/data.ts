// Fetch helpers and hooks for the stress page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import type { StressModel, StressSummary, TryRequest, TryResponse, Intervention } from './types';

const SUMMARY_URL = '/api/stress/summary';
const MODELS_URL = '/api/stress/models';
const INTERVENTIONS_URL = '/api/stress/interventions';
const TRY_URL = '/api/stress/try';

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

/** Saved results: stress tests, gate interventions and write-gate patches. */
export function useStressSummary() {
  return useRetrying<StressSummary>(SUMMARY_URL, () => request<StressSummary>(SUMMARY_URL));
}

/** The local overwrite checkpoints this machine can run. */
export function useStressModels() {
  return useRetrying<StressModel[]>(MODELS_URL, () => request<StressModel[]>(MODELS_URL));
}

export function useInterventions() {
  return useRetrying<Intervention[]>(INTERVENTIONS_URL, () => request<Intervention[]>(INTERVENTIONS_URL));
}

export function postTry(body: TryRequest): Promise<TryResponse> {
  return request<TryResponse>(TRY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** Live run. Debounced; the last settled request wins. Pass null to skip. */
export function useTry(body: TryRequest | null, delay = 220) {
  const key = body === null ? null : JSON.stringify(body);
  return useRetrying<TryResponse>(key, () => postTry(JSON.parse(String(key)) as TryRequest), delay);
}

// Fetch helpers and hooks for the decay-initialization page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import {
  GATES_URL,
  MODELS_URL,
  SUMMARY_URL,
  TRY_URL,
  type Gates,
  type LocalModel,
  type Summary,
  type TryRequest,
  type TryResponse,
} from './types';

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

export function useDecaySummary() {
  return useRetrying<Summary>(SUMMARY_URL, () => request<Summary>(SUMMARY_URL));
}

export function useDecayGates() {
  return useRetrying<Gates>(GATES_URL, () => request<Gates>(GATES_URL));
}

export function useLocalDecayModels() {
  return useRetrying<LocalModel[]>(MODELS_URL, () => request<LocalModel[]>(MODELS_URL));
}

export function postTry(body: TryRequest): Promise<TryResponse> {
  return request<TryResponse>(TRY_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** Live example. Debounced by the caller through the key; the last settled request wins. Pass null to skip. */
export function useTry(body: TryRequest | null, delay = 220) {
  const key = body === null ? null : JSON.stringify(body);
  return useRetrying<TryResponse>(key, () => postTry(JSON.parse(String(key)) as TryRequest), delay);
}

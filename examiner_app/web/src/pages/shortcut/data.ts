// Fetch helpers and hooks for the shortcut page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import {
  EVALUATE_URL,
  MODELS_URL,
  SUMMARY_URL,
  type EvaluateRequest,
  type EvaluateResponse,
  type ShortcutModel,
  type ShortcutSummary,
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

/** Summary of the saved block-layout confirmation (hypotheses, per-model results, patches). */
export function useShortcutSummary() {
  return useRetrying<ShortcutSummary>(SUMMARY_URL, () => request<ShortcutSummary>(SUMMARY_URL));
}

/** The local overwrite checkpoints this machine can run. */
export function useLocalModels() {
  return useRetrying<ShortcutModel[]>(MODELS_URL, () => request<ShortcutModel[]>(MODELS_URL));
}

export function postEvaluate(body: EvaluateRequest): Promise<EvaluateResponse> {
  return request<EvaluateResponse>(EVALUATE_URL, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
}

/** Live evaluation. Debounced; the last settled request wins. Pass null to skip. */
export function useEvaluate(body: EvaluateRequest | null, delay = 220) {
  const key = body === null ? null : JSON.stringify(body);
  return useRetrying<EvaluateResponse>(key, () => postEvaluate(JSON.parse(String(key)) as EvaluateRequest), delay);
}

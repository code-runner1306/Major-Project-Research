// Fetch hooks for the attention control page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import { EXAMPLE_URL, SUMMARY_URL, type AttentionControlSummary, type OverwriteExample } from './types';

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

/** One GET. A network failure becomes an ApiFailure with status null, which useRetrying treats as "server starting". */
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

export function useAttentionSummary() {
  return useRetrying<AttentionControlSummary>(SUMMARY_URL, () => request<AttentionControlSummary>(SUMMARY_URL));
}

/** One real overwrite example from the diagnostic's generator (held-out split). */
export function useOverwriteExample(index = 0) {
  const url = `${EXAMPLE_URL}?index=${index}`;
  return useRetrying<OverwriteExample>(url, () => request<OverwriteExample>(url));
}

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const pct2 = (v: number) => pct(v, 2);

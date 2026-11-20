// Fetch hooks for the Anatomy page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import { EXAMPLE_URL, RUNS_URL, SUMMARY_URL, type AnatomyExample, type RunRow, type Summary } from './types';

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
async function getJson<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiFailure('The server did not answer.', null);
  }
  if (!res.ok) throw new ApiFailure(await readDetail(res), res.status);
  return (await res.json()) as T;
}

/** Saved results of the three experiments: per-seed accuracy, tests and mechanism runs. */
export function useAnatomySummary() {
  return useRetrying<Summary>(SUMMARY_URL, () => getJson<Summary>(SUMMARY_URL));
}

/** Every saved run as a flat table (optionally filtered by the server). */
export function useAnatomyRuns() {
  return useRetrying<RunRow[]>(RUNS_URL, () => getJson<RunRow[]>(RUNS_URL));
}

/** One example from the real Anatomy generator for an experiment and seed. */
export function useAnatomyExample(experiment: 'E4' | 'E5', seed: number) {
  const url = `${EXAMPLE_URL}?experiment=${experiment}&seed=${seed}`;
  return useRetrying<AnatomyExample>(url, () => getJson<AnatomyExample>(url), 150);
}

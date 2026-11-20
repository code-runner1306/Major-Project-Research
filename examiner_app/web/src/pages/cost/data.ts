// Fetch hooks for the cost page. Retrying while the server starts is shared with the dataset page.
import { ApiFailure, useRetrying } from '../dataset/shared';
import type { CostSummary, ParametersResponse } from './types';

export const SUMMARY_URL = '/api/cost/summary';
export const PARAMETERS_URL = '/api/cost/parameters';

async function request<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiFailure('The server did not answer.', null);
  }
  if (!res.ok) throw new ApiFailure(`${res.status} ${res.statusText}`.trim(), res.status);
  return (await res.json()) as T;
}

/** Saved timing, projections and the not-recorded list (one file, built by server/build_cost_export.py). */
export function useCostSummary() {
  return useRetrying<CostSummary>(SUMMARY_URL, () => request<CostSummary>(SUMMARY_URL));
}

/** Parameter counts and per-layer gate sizes, counted live from the recipe model constructors on CPU. */
export function useCostParameters() {
  return useRetrying<ParametersResponse>(PARAMETERS_URL, () => request<ParametersResponse>(PARAMETERS_URL));
}

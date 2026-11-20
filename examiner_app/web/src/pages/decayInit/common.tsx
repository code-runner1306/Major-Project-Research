import type { ReactNode } from 'react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import type { DecayInit, Granularity } from './types';

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** A fraction difference shown in percentage points, with a true minus sign. */
export function pointsSigned(v: number, digits = 1): string {
  const x = v * 100;
  const sign = x > 0 ? '+' : x < 0 ? '−' : '';
  return `${sign}${Math.abs(x).toFixed(digits)} pts`;
}

export const CONDITION_LABEL: Record<string, string> = {
  'default/scalar': 'Scalar, default start',
  'default/channel-wise': 'Channel-wise, default start',
  'mean-decay/scalar': 'Scalar, mean-decay start',
  'mean-decay/channel-wise': 'Channel-wise, mean-decay start',
  'joint/scalar': 'Scalar, joint start',
  'joint/channel-wise': 'Channel-wise, joint start',
};

export function conditionLabel(init: DecayInit, variant: Granularity): string {
  return CONDITION_LABEL[`${init}/${variant}`];
}

export interface RemoteState<T> {
  data: T | null;
  error: { message: string; status: number | null } | null;
  waiting: boolean;
  pending: boolean;
}

/** Skeleton while loading, retry notice while the server starts, error panel on failure, dimmed old data while a new request runs. */
export function Remote<T>({
  state,
  what,
  tall = false,
  children,
}: {
  state: RemoteState<T>;
  what: string;
  tall?: boolean;
  children: (data: T) => ReactNode;
}) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={`loading ${what}`} tall={tall} />;
  return <div className={state.pending ? 'di-remote di-pending' : 'di-remote'}>{children(state.data)}</div>;
}

/** Segmented choice (two to five options). */
export function Segmented<T extends string | number>({
  options,
  value,
  onChange,
  label,
}: {
  options: { id: T; label: string }[];
  value: T;
  onChange: (v: T) => void;
  label: string;
}) {
  return (
    <div className="di-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.id)} type="button" aria-pressed={value === o.id} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/**
 * Average decay as a memory half-life in tokens: the number of tokens after which a value written now keeps half
 * its weight, if the decay stayed at this average. A diagnostic of the gate, not a measured memory lifetime.
 */
export function halfLifeTokens(mean: number): number {
  return -1 / Math.log2(mean);
}

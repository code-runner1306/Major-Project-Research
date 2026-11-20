import type { ReactNode } from 'react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;
export const pctAxis = (v: number) => `${Math.round(v)}%`;

export interface RemoteState<T> {
  data: T | null;
  error: { message: string; status: number | null } | null;
  waiting: boolean;
  pending: boolean;
}

/** Skeleton while loading, retry notice while the server starts, error panel on failure. */
export function Remote<T>({ state, what, tall = false, children }: { state: RemoteState<T>; what: string; tall?: boolean; children: (data: T) => ReactNode }) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={`loading ${what}`} tall={tall} />;
  return <div className={state.pending ? 'ctl-remote ctl-pending' : 'ctl-remote'}>{children(state.data)}</div>;
}

/** Segmented choice (two to five options). */
export function Segmented<T extends string | number>({ options, value, onChange, label }: { options: { id: T; label: string }[]; value: T; onChange: (v: T) => void; label: string }) {
  return (
    <div className="ctl-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.id)} type="button" aria-pressed={value === o.id} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

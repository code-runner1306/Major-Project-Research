import type { ReactNode } from 'react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import { modelKey, modelLabel, sameModel, type LocalModel, type ModelRef } from './types';

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
  label,
  children,
}: {
  state: RemoteState<T>;
  what: string;
  tall?: boolean;
  label?: string;
  children: (data: T) => ReactNode;
}) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={label ?? `loading ${what}`} tall={tall} />;
  return <div className={state.pending ? 'ln-remote ln-pending' : 'ln-remote'}>{children(state.data)}</div>;
}

/** Dropdown of local models. Options are the models this machine can run. */
export function ModelSelect({
  models,
  value,
  onChange,
  label = 'Model',
}: {
  models: LocalModel[];
  value: ModelRef;
  onChange: (m: ModelRef) => void;
  label?: string;
}) {
  return (
    <label className="ln-field">
      <span className="ln-field__label">{label}</span>
      <select
        value={modelKey(value)}
        onChange={(e) => {
          const next = models.find((m) => modelKey(m) === e.target.value);
          if (next) onChange({ init: next.init, variant: next.variant, seed: next.seed });
        }}
      >
        {!models.some((m) => sameModel(m, value)) && <option value={modelKey(value)}>{modelLabel(value)}</option>}
        {models.map((m) => (
          <option key={modelKey(m)} value={modelKey(m)}>
            {modelLabel(m)}
          </option>
        ))}
      </select>
    </label>
  );
}

export function LocalBadge({ local }: { local: boolean }) {
  return local ? (
    <span className="ln-badge is-local">local</span>
  ) : (
    <span className="ln-badge is-unavailable">Models unavailable</span>
  );
}

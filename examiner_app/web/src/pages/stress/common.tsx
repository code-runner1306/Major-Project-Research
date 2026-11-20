import { useEffect, useRef, type ReactNode } from 'react';
import { animate, motion, useReducedMotion } from 'motion/react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import { pct, pointsSigned } from './derive';
import type { StressModel } from './types';
import { settingLabel } from './derive';

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
  return <div className={state.pending ? 'stress-remote stress-pending' : 'stress-remote'}>{children(state.data)}</div>;
}

/** Segmented choice (two to six options). */
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
    <div className="stress-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.id)} type="button" aria-pressed={value === o.id} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Dropdown of local checkpoints. */
export function ModelSelect({
  models,
  value,
  onChange,
  compact = false,
}: {
  models: StressModel[];
  value: { init: string; variant: StressModel['variant']; seed: number };
  onChange: (m: { init: string; variant: StressModel['variant']; seed: number }) => void;
  /** Drops the visible label; the select keeps the name "Model" for screen readers. */
  compact?: boolean;
}) {
  const key = (m: { init: string; variant: string; seed: number }) => `${m.init}|${m.variant}|${m.seed}`;
  return (
    <label className="stress-field">
      {!compact && <span className="stress-field__label">Model</span>}
      <select
        aria-label="Model"
        value={key(value)}
        onChange={(e) => {
          const next = models.find((m) => key(m) === e.target.value);
          if (next) onChange(next);
        }}
      >
        {models.map((m) => (
          <option key={key(m)} value={key(m)}>
            {`${settingLabel(m.init, m.variant)} · seed ${m.seed}`}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Number input with a plain label. */
export function NumberField({
  label,
  value,
  min,
  max,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  onChange: (v: number) => void;
}) {
  return (
    <label className="stress-field">
      <span className="stress-field__label">{label}</span>
      <input
        type="number"
        value={value}
        min={min}
        max={max}
        onChange={(e) => {
          const next = Math.round(Number(e.target.value));
          if (Number.isFinite(next)) onChange(Math.min(max, Math.max(min, next)));
        }}
      />
    </label>
  );
}

/** One bar that tweens to its value when the value changes. */
function Fill({ value, className, reduce }: { value: number; className: string; reduce: boolean }) {
  const width = `${Math.min(1, Math.max(0, value)) * 100}%`;
  return (
    <motion.div
      className={className}
      initial={{ width: '0%' }}
      animate={{ width }}
      transition={reduce ? { duration: 0 } : { duration: 0.6, ease: 'easeOut' }}
    />
  );
}

/** Before and after bars for each measure, on the same 0 to 100% scale. Tweens when the values change. */
export function PairBars({
  rows,
}: {
  rows: { label: string; before: number; after: number }[];
}) {
  const reduce = useReducedMotion() ?? false;
  return (
    <div className="stress-pair">
      {rows.map((r) => (
        <div className="stress-pair__row" key={r.label}>
          <div className="stress-pair__label">{r.label}</div>
          <div className="stress-pair__bars">
            <div className="stress-pair__track">
              <Fill value={r.before} className="stress-pair__fill is-before" reduce={reduce} />
            </div>
            <div className="stress-pair__track">
              <Fill value={r.after} className="stress-pair__fill is-after" reduce={reduce} />
            </div>
          </div>
          <div className="stress-pair__values">
            <span className="is-before">{pct(r.before)}</span>
            <span className="is-after">{pct(r.after)}</span>
          </div>
        </div>
      ))}
    </div>
  );
}

/** Horizontal bars around a zero line: positive grows right, negative grows left. Tweens on change. */
export function SignedBars({
  rows,
  range,
}: {
  rows: { label: string; value: number; note: string }[];
  range: number;
}) {
  const reduce = useReducedMotion() ?? false;
  return (
    <div className="stress-signed">
      {rows.map((r) => {
        const half = Math.min(1, Math.abs(r.value) / range) * 50;
        const left = r.value >= 0 ? 50 : 50 - half;
        return (
          <div className="stress-signed__row" key={r.label}>
            <div className="stress-signed__label">
              {r.label}
              <span className="stress-signed__note">{r.note}</span>
            </div>
            <div className="stress-signed__track">
              <motion.div
                className={r.value >= 0 ? 'stress-signed__bar is-up' : 'stress-signed__bar is-down'}
                initial={false}
                animate={{ left: `${left}%`, width: `${half}%` }}
                transition={reduce ? { duration: 0 } : { duration: 0.6, ease: 'easeOut' }}
              />
              <span className="stress-signed__zero" />
            </div>
            <div className="stress-signed__value">{pointsSigned(r.value)}</div>
          </div>
        );
      })}
    </div>
  );
}

/** Number that counts from its previous value to `value`. Static under reduced motion. */
export function CountUp({ value, format }: { value: number; format: (v: number) => string }) {
  const reduce = useReducedMotion() ?? false;
  const ref = useRef<HTMLSpanElement>(null);
  const fromRef = useRef(value);
  useEffect(() => {
    if (reduce) {
      fromRef.current = value;
      if (ref.current) ref.current.textContent = format(value);
      return;
    }
    const controls = animate(fromRef.current, value, {
      duration: 0.9,
      ease: 'easeOut',
      onUpdate: (v) => {
        fromRef.current = v;
        if (ref.current) ref.current.textContent = format(v);
      },
    });
    return () => controls.stop();
    // format is a stable module-level function at every call site.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value, reduce]);
  return <span ref={ref}>{format(value)}</span>;
}

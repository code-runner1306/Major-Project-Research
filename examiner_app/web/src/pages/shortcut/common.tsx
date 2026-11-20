import { useEffect, useRef, useState, type ReactNode } from 'react';
import { animate, motion, useInView, useReducedMotion } from 'motion/react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import { modelKey, modelLabel, sameModel, type ModelRef } from './types';
import type { Variant } from '../../api';

export const pct = (v: number, digits = 1) => `${(v * 100).toFixed(digits)}%`;

/** A fraction difference shown in percentage points, with a sign. */
export function pointsSigned(v: number, digits = 1): string {
  const x = v * 100;
  const sign = x > 0 ? '+' : x < 0 ? '−' : '';
  return `${sign}${Math.abs(x).toFixed(digits)} pts`;
}

/**
 * Display text for a value of unknown shape. Strings are shown verbatim, whole numbers stay whole, other finite
 * numbers become percentages (0 to 1) or two decimals. Anything else (objects, null, NaN) shows an em dash, so this never throws.
 */
export function formatValue(v: unknown): string {
  if (typeof v === 'string') return v;
  if (typeof v === 'number' && Number.isFinite(v)) {
    if (Number.isInteger(v)) return String(v);
    if (v >= 0 && v <= 1) return pct(v);
    return v.toFixed(2);
  }
  return '—';
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
  return <div className={state.pending ? 'sc-remote sc-pending' : 'sc-remote'}>{children(state.data)}</div>;
}

/** Number that counts from its previous value to `value` whenever it changes. Static under reduced motion. */
export function CountUp({
  value,
  format = (v: number) => pct(v),
  active = true,
  duration = 0.9,
}: {
  value: number;
  format?: (v: number) => string;
  active?: boolean;
  duration?: number;
}) {
  const reduce = useReducedMotion() ?? false;
  const [shown, setShown] = useState(reduce ? value : 0);
  const fromRef = useRef(reduce ? value : 0);

  useEffect(() => {
    if (!active) return;
    if (reduce) {
      fromRef.current = value;
      setShown(value);
      return;
    }
    const controls = animate(fromRef.current, value, {
      duration,
      ease: 'easeOut',
      onUpdate: (v) => {
        fromRef.current = v;
        setShown(v);
      },
    });
    return () => controls.stop();
  }, [value, active, reduce, duration]);

  return <>{format(shown)}</>;
}

/** Pass or fail stamp that drops onto the page when scrolled into view. */
export function Stamp({ pass, label, delay = 0 }: { pass: boolean; label: string; delay?: number }) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion() ?? false;
  const shown = reduce || inView;
  return (
    <motion.span
      ref={ref}
      className={pass ? 'sc-stamp is-pass' : 'sc-stamp is-fail'}
      initial={false}
      animate={shown ? { opacity: 1, scale: 1, rotate: -6 } : { opacity: 0, scale: 1.7, rotate: -14 }}
      transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 260, damping: 13, delay }}
    >
      {label}
    </motion.span>
  );
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
    <div className="sc-seg" role="group" aria-label={label}>
      {options.map((o) => (
        <button key={String(o.id)} type="button" aria-pressed={value === o.id} onClick={() => onChange(o.id)}>
          {o.label}
        </button>
      ))}
    </div>
  );
}

/** Dropdown of local models. Keys are the model's init, variant and seed. */
export function ModelSelect({
  models,
  value,
  onChange,
  label = 'Model',
}: {
  models: ModelRef[];
  value: ModelRef;
  onChange: (m: ModelRef) => void;
  label?: string;
}) {
  const current = models.find((m) => sameModel(m, value)) ?? value;
  const options = models.some((m) => sameModel(m, value)) ? models : [value, ...models];
  return (
    <label className="sc-field">
      <span className="sc-field__label">{label}</span>
      <select
        value={modelKey(current)}
        onChange={(e) => {
          const next = options.find((m) => modelKey(m) === e.target.value);
          if (next) onChange(next);
        }}
      >
        {options.map((m) => (
          <option key={modelKey(m)} value={modelKey(m)}>
            {modelLabel(m)}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Short badge text for a local or unavailable model. */
export function ModelBadge({ local }: { local: boolean }) {
  return local ? (
    <span className="sc-badge is-local">local</span>
  ) : (
    <span className="sc-badge is-unavailable">Models unavailable</span>
  );
}

export function variantText(v: Variant): string {
  return v === 'channel-wise' ? 'channel-wise' : 'scalar';
}

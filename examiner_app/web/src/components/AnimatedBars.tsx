import { useEffect, useRef, useState } from 'react';
import { animate, motion, useInView, useReducedMotion } from 'motion/react';
import './viz.css';

const ACCENT = 'var(--accent)';

export interface BarDatum {
  label: string;
  value: number;
  color?: string;
  note?: string;
}

export interface AnimatedBarsProps {
  /** Horizontal bars that grow from zero when scrolled into view and tween when the data changes. */
  data: BarDatum[];
  /** Value that maps to a full-width bar (default: the largest value). */
  max?: number;
  /** Formats the counting value label (default: up to 3 decimals). */
  format?: (value: number) => string;
}

const defaultFormat = (v: number) => v.toLocaleString(undefined, { maximumFractionDigits: 3 });

/** Value label that counts up from its previous value whenever `target` changes while `active`. */
function CountUp({
  target,
  active,
  format,
}: {
  target: number;
  active: boolean;
  format: (value: number) => string;
}) {
  const reduce = useReducedMotion() ?? false;
  const [shown, setShown] = useState(0);
  const fromRef = useRef(0);

  useEffect(() => {
    if (!active) return;
    if (reduce) {
      fromRef.current = target;
      setShown(target);
      return;
    }
    const controls = animate(fromRef.current, target, {
      duration: 0.9,
      ease: 'easeOut',
      onUpdate: (v) => {
        fromRef.current = v;
        setShown(v);
      },
    });
    return () => controls.stop();
  }, [target, active, reduce]);

  return <span className="viz-bars__value">{format(shown)}</span>;
}

/** Horizontal animated bar chart: grows on view, tweens on data change, counts up value labels. */
export function AnimatedBars({ data, max, format = defaultFormat }: AnimatedBarsProps) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.25 });
  const reduce = useReducedMotion() ?? false;

  const top = max ?? Math.max(1e-9, ...data.map((d) => d.value));

  return (
    <div className="viz-bars" ref={ref}>
      {data.map((d, i) => {
        const pct = Math.min(1, Math.max(0, d.value / top)) * 100;
        return (
          <div className="viz-bars__row" key={`${i}:${d.label}`}>
            <div className="viz-bars__label" title={d.note}>
              {d.label}
              {d.note && <span className="viz-bars__note">{d.note}</span>}
            </div>
            <div className="viz-bars__track">
              <motion.div
                className="viz-bars__fill"
                style={{ background: d.color ?? ACCENT }}
                initial={{ width: '0%' }}
                animate={{ width: inView ? `${pct}%` : '0%' }}
                transition={
                  reduce
                    ? { duration: 0 }
                    : { type: 'spring', stiffness: 70, damping: 18, delay: inView ? i * 0.06 : 0 }
                }
              />
            </div>
            <CountUp target={d.value} active={inView} format={format} />
          </div>
        );
      })}
    </div>
  );
}

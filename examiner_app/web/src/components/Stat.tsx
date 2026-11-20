import { useEffect, useRef, useState, type ReactNode } from 'react';
import { animate, useReducedMotion } from 'motion/react';
import './story.css';

const asPercent = (v: number) => `${Math.round(v * 100)}%`;

/** Big number with a small label. Counts from its previous value to `value`. Static under reduced motion. */
export function Stat({
  value,
  label,
  format = asPercent,
  tone = 'neutral',
}: {
  value: number;
  label: ReactNode;
  format?: (v: number) => string;
  tone?: 'neutral' | 'ok' | 'bad';
}) {
  const reduce = useReducedMotion() ?? false;
  const [shown, setShown] = useState(reduce ? value : 0);
  const fromRef = useRef(reduce ? value : 0);

  useEffect(() => {
    if (reduce) {
      fromRef.current = value;
      setShown(value);
      return;
    }
    const controls = animate(fromRef.current, value, {
      duration: 0.9,
      ease: 'easeOut',
      onUpdate: (v) => {
        fromRef.current = v;
        setShown(v);
      },
    });
    return () => controls.stop();
  }, [value, reduce]);

  return (
    <div className={`stat is-${tone}`}>
      <div className="stat__value">{format(shown)}</div>
      <div className="stat__label">{label}</div>
    </div>
  );
}

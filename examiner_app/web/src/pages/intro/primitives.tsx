import { useEffect, useRef, useState, type ReactNode } from 'react';
import { animate, motion, useInView, useReducedMotion } from 'motion/react';

const defaultInt = (n: number) => Math.round(n).toLocaleString('en-US');

/** Fades and lifts its children in once they scroll into view. `index` staggers siblings. */
export function Reveal({ children, index = 0, className }: { children: ReactNode; index?: number; className?: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.2 });
  const reduce = useReducedMotion() ?? false;
  const shown = inView || reduce;
  return (
    <motion.div
      ref={ref}
      className={className}
      initial={false}
      animate={shown ? { opacity: 1, y: 0 } : { opacity: 0, y: 16 }}
      transition={{ duration: reduce ? 0 : 0.55, ease: 'easeOut', delay: reduce ? 0 : Math.min(index, 8) * 0.08 }}
    >
      {children}
    </motion.div>
  );
}

/** Number that counts up from 0 to `to` the first time it scrolls into view. */
export function CountUp({
  to,
  format = defaultInt,
  className,
}: {
  to: number;
  format?: (n: number) => string;
  className?: string;
}) {
  const ref = useRef<HTMLSpanElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion() ?? false;
  const [shown, setShown] = useState(reduce ? to : 0);

  useEffect(() => {
    if (!inView) return;
    if (reduce) {
      setShown(to);
      return;
    }
    const controls = animate(0, to, {
      duration: 1.2,
      ease: 'easeOut',
      onUpdate: (v) => setShown(v),
    });
    return () => controls.stop();
  }, [inView, to, reduce]);

  return (
    <span ref={ref} className={className}>
      {format(shown)}
    </span>
  );
}

/** Loading placeholder: pulsing grey bars (the pulse is disabled by the CSS reduced-motion rule). */
export function Skeleton({ lines = 3, label }: { lines?: number; label: string }) {
  return (
    <div className="intro-skeleton" role="status" aria-busy="true" aria-label={label}>
      {Array.from({ length: lines }, (_, i) => (
        <span key={i} className="intro-skeleton__bar" style={{ width: `${100 - i * 16}%` }} />
      ))}
    </div>
  );
}

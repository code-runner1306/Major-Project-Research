import { useRef } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { Stamp, formatValue, pct } from './common';
import { WhatThisShows } from '../../components/WhatThisShows';
import { MoreDetail } from '../../components/MoreDetail';
import type { Hypothesis } from './types';

/** Plain statement for each hypothesis. Protocol identifiers stay out of visible text. */
const PLAIN: Record<string, string> = {
  H1: 'Most models show the shortcut',
  H2: 'Most wrong answers are the first replacement',
  H3: 'Models pass the ordinary layout',
  H4: 'Only the exact block-first layout fails',
  H5: 'The gap persists with more filler',
};

/** A number clamped to [0, 1], or null when the server sent something that is not a finite number. */
function fraction(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? Math.min(1, Math.max(0, v)) : null;
}

/** Observed bar that grows from 0 when scrolled into view, with a marker at the threshold set in advance. */
function ScoreBar({ observed, threshold }: { observed: number | null; threshold: number | null }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.6 });
  const reduce = useReducedMotion() ?? false;
  if (observed === null) return null;
  const fill = inView || reduce ? `${observed * 100}%` : '0%';
  return (
    <div className="sc-bar" ref={ref} role="img" aria-label={`Observed ${pct(observed)}${threshold === null ? '' : `, threshold ${pct(threshold)}`}`}>
      <motion.div
        className="sc-bar__fill"
        initial={false}
        animate={{ width: fill }}
        transition={reduce ? { duration: 0 } : { duration: 0.9, ease: 'easeOut' }}
      />
      {threshold !== null && <span className="sc-bar__marker" style={{ left: `${threshold * 100}%` }} />}
    </div>
  );
}

/** Five checks, each with a plain statement, what was observed, a bar against its threshold, and a stamp. */
export function Scoreboard({ hypotheses }: { hypotheses: Record<string, Hypothesis> }) {
  const ids = Object.keys(hypotheses).sort();
  return (
    <div className="sc-score-wrap">
      <ol className="sc-score">
        {ids.map((id, i) => {
          const h = hypotheses[id];
          const observed = fraction(h.observed_value);
          const threshold = fraction(h.threshold_value);
          return (
            <li key={id} className="sc-score__row">
              <span className="sc-score__id">{i + 1}</span>
              <div className="sc-score__body">
                <p className="sc-score__statement">{PLAIN[id] ?? 'Hypothesis'}</p>
                <p className="sc-score__observed">{formatValue(h.observed_text)}</p>
              </div>
              <ScoreBar observed={observed} threshold={threshold} />
              <div className="sc-score__side">
                <Stamp pass={h.confirmed} label={h.confirmed ? 'PASS' : 'FAIL'} delay={i * 0.15} />
                <MoreDetail>
                  <p>{formatValue(h.statement)}</p>
                  <p>Threshold set in advance: {formatValue(h.threshold)}</p>
                </MoreDetail>
              </div>
            </li>
          );
        })}
      </ol>
      <WhatThisShows>bars show what was observed; the tick marks the threshold set in advance.</WhatThisShows>
    </div>
  );
}

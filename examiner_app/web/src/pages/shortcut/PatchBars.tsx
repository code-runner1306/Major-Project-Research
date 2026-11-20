import { useRef } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { WhatThisShows } from '../../components/WhatThisShows';
import { pointsSigned } from './common';
import type { PatchRow } from './types';

const PATCH_NAMES: Record<string, string> = {
  beta_one_latest: 'Write strength 1 on the latest write',
  key_align_latest: 'Earlier writes use the latest key',
  beta_zero_rewrite1: 'Switch off the first replacement',
  beta_copy_rewrite1: 'Later writes copy the first replacement’s strength',
  control_beta_one_unrelated: 'Control: write strength 1 on an unrelated write',
};

/** A bar that grows left or right from a centre line. value and scale are fractions; scale is the length of a half-track. */
export function SignedBar({ value, scale, label }: { value: number; scale: number; label: string }) {
  const ref = useRef<HTMLDivElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.5 });
  const reduce = useReducedMotion() ?? false;
  const shown = inView || reduce;
  const len = Math.min(50, (Math.abs(value) / scale) * 50);
  const left = value >= 0 ? 50 : 50 - len;

  return (
    <div className="sc-signed-row">
      <span className="sc-signed-row__label">{label}</span>
      <div className="sc-signed" ref={ref} role="img" aria-label={`${label}: ${pointsSigned(value)}`}>
        <span className="sc-signed__axis" />
        <motion.span
          className={value < 0 ? 'sc-signed__bar is-down' : 'sc-signed__bar is-up'}
          initial={{ left: '50%', width: '0%' }}
          animate={{ left: shown ? `${left}%` : '50%', width: shown ? `${len}%` : '0%' }}
          transition={reduce ? { duration: 0 } : { duration: 0.8, ease: 'easeOut' }}
        />
      </div>
      <span className="sc-signed-row__value">{pointsSigned(value)}</span>
    </div>
  );
}

/** Section 4c: what each discovery-stage patch did to the three measures, on the same fresh layouts. */
export function PatchBars({ patches }: { patches: PatchRow[] }) {
  const scale = Math.max(
    0.005,
    ...patches.flatMap((p) => [Math.abs(p.latest_delta), Math.abs(p.unrelated_delta), Math.abs(p.first_rewrite_delta)]),
  );

  return (
    <>
      <div className="sc-patches">
        {patches.map((p) => (
          <article className="sc-card" key={p.patch}>
            <h4 className="sc-h4">{PATCH_NAMES[p.patch] ?? p.patch}</h4>
            <p className="sc-muted sc-card__note">
              Latest value up in {p.models_up} models, down in {p.models_down}.
            </p>
            <SignedBar value={p.latest_delta} scale={scale} label="Latest value" />
            <SignedBar value={p.unrelated_delta} scale={scale} label="Unrelated key" />
            <SignedBar value={p.first_rewrite_delta} scale={scale} label="First replacement" />
          </article>
        ))}
      </div>
      <WhatThisShows>each card is one patch. Bars right mean the score went up; bars left, it went down.</WhatThisShows>
    </>
  );
}

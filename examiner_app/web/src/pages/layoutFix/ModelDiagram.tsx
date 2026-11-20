import { useRef } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { useApi } from '../../components/useApi';
import type { LayoutFixSummary } from '../../api';
import { WhatThisShows } from '../../components/WhatThisShows';

interface RunRow {
  arm: string;
  variant: string;
  seed: number;
  path: string;
  final_dev_accuracy: number;
}

const INK = 'var(--text)';
const LINE = 'var(--border-strong)';

function Box({
  x,
  y,
  w,
  h,
  title,
  sub,
  accent,
  delay,
  show,
}: {
  x: number;
  y: number;
  w: number;
  h: number;
  title: string;
  sub?: string;
  accent?: boolean;
  delay: number;
  show: boolean;
}) {
  const reduce = useReducedMotion() ?? false;
  return (
    <motion.g
      initial={reduce ? false : { opacity: 0, y: 6 }}
      animate={{ opacity: show ? 1 : 0, y: show ? 0 : 6 }}
      transition={{ duration: reduce ? 0 : 0.5, delay: reduce ? 0 : delay, ease: 'easeOut' }}
    >
      <rect x={x} y={y} width={w} height={h} rx={8} style={{ fill: accent ? 'color-mix(in srgb, var(--accent) 10%, var(--surface))' : 'var(--surface)', stroke: accent ? 'var(--accent)' : LINE }} strokeWidth={1.4} />
      <text x={x + w / 2} y={y + (sub ? h / 2 - 2 : h / 2 + 5)} textAnchor="middle" fontSize={14} fontWeight={600} style={{ fill: INK }}>
        {title}
      </text>
      {sub && (
        <text x={x + w / 2} y={y + h / 2 + 15} textAnchor="middle" fontSize={11} style={{ fill: 'var(--text-muted)' }}>
          {sub}
        </text>
      )}
    </motion.g>
  );
}

function Arrow({ d, delay, show }: { d: string; delay: number; show: boolean }) {
  const reduce = useReducedMotion() ?? false;
  return (
    <motion.path
      d={d}
      style={{ fill: 'none', stroke: LINE }}
      strokeWidth={1.5}
      markerEnd="url(#lf-arrow)"
      initial={reduce ? false : { pathLength: 0 }}
      animate={{ pathLength: show || reduce ? 1 : 0 }}
      transition={{ duration: reduce ? 0 : 0.6, delay: reduce ? 0 : delay, ease: 'easeInOut' }}
    />
  );
}

/** Animated block diagram of the model. The same architecture is used in every arm. */
export function ModelDiagram() {
  const ref = useRef<SVGSVGElement>(null);
  const show = useInView(ref, { once: true, amount: 0.3 });

  return (
    <div className="lf-model">
      <svg ref={ref} className="lf-model__svg" viewBox="0 0 920 290" role="img" aria-label="Model: tokens, embedding, four layers of short convolution, Gated DeltaNet memory and output, then prediction">
        <defs>
          <marker id="lf-arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M0,0 L10,5 L0,10 z" style={{ fill: LINE }} />
          </marker>
        </defs>

        <Box x={8} y={110} w={112} h={60} title="Tokens" sub="ids in, one at a time" delay={0} show={show} />
        <Box x={150} y={110} w={112} h={60} title="Embedding" sub="id to vector" delay={0.15} show={show} />
        <Arrow d="M120,140 L150,140" delay={0.35} show={show} />

        <motion.g
          initial={{ opacity: 0 }}
          animate={{ opacity: show ? 1 : 0 }}
          transition={{ duration: 0.6, delay: 0.5 }}
        >
          <rect x={290} y={16} width={386} height={258} rx={12} style={{ fill: 'var(--surface-2)', stroke: LINE }} strokeDasharray="5 4" />
          <text x={306} y={38} fontSize={13} fontWeight={600} style={{ fill: INK }}>
            × 4 layers, same design in each
          </text>
        </motion.g>
        <Arrow d="M262,140 L290,140" delay={0.6} show={show} />

        <Box x={310} y={56} w={346} h={44} title="Short convolution" sub="mixes each token with its neighbours" delay={0.7} show={show} />
        <Arrow d="M483,100 L483,118" delay={0.95} show={show} />
        <Box x={310} y={118} w={346} h={68} title="Gated DeltaNet memory" sub="decay gate α (forget) · write gate β (write)" accent delay={1.0} show={show} />
        <Arrow d="M483,186 L483,204" delay={1.25} show={show} />
        <Box x={310} y={204} w={346} h={44} title="Output" sub="read the memory for each token" delay={1.3} show={show} />

        <Arrow d="M676,140 L712,140" delay={1.6} show={show} />
        <Box x={712} y={110} w={150} h={60} title="Prediction" sub="next value" delay={1.7} show={show} />
      </svg>
      <WhatThisShows>data flows left to right. All arms share this design and starting weights.</WhatThisShows>
    </div>
  );
}

/** Parameter counts and run counts, all read from the API. */
export function ModelSizes({ s }: { s: LayoutFixSummary | null }) {
  const runs = useApi<RunRow[]>('/api/runs');
  const scalarP = s ? s.gates.scalar.parameters.toLocaleString() : '…';
  const channelP = s ? s.gates['channel-wise'].parameters.toLocaleString() : '…';

  return (
    <div className="lf-model">
      <div className="lf-model__facts">
        <div>
          <strong>Scalar decay</strong>
          <span>{scalarP} parameters</span>
          <p>One forgetting rate per layer and step.</p>
        </div>
        <div>
          <strong>Channel-wise decay</strong>
          <span>{channelP} parameters</span>
          <p>One forgetting rate per memory column.</p>
        </div>
        <div>
          <strong>Trained runs</strong>
          <span>
            {runs.loading && 'checking…'}
            {runs.error && 'run list unavailable (start the server)'}
            {runs.data && `${runs.data.length} layout-fix runs`}
          </span>
          <p>{s ? s.curves.length : '…'} models: five arms, two decay types, three seeds.</p>
        </div>
      </div>
      <WhatThisShows>only the training data differs between the trained models.</WhatThisShows>
    </div>
  );
}

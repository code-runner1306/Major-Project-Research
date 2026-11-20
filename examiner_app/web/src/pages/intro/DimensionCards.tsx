import type { LayoutFixSummary, Variant } from '../../api';
import { AnimatedBars } from '../../components/AnimatedBars';
import { Term } from '../../components/Term';
import { WhatThisShows } from '../../components/WhatThisShows';
import type { ApiResult } from '../../components/useApi';
import { CountUp, Reveal, Skeleton } from './primitives';

const SCALAR = 'var(--c-scalar)';
const CHANNEL = 'var(--c-channel)';
const TRACE = 'A4_distance_and_rewrites/scalar';

const shape = (dims: number[]) => `[${dims.join(', ')}]`;
const intFmt = (n: number) => Math.round(n).toLocaleString('en-US');

function GranularityCard({
  variant,
  colour,
  summary,
  index,
}: {
  variant: Variant;
  colour: string;
  summary: LayoutFixSummary;
  index: number;
}) {
  const gate = summary.gates[variant];
  const layer = gate.layers[0];
  const perToken = layer.alpha_shape[layer.alpha_shape.length - 1];
  const memory = summary.delta_rule[TRACE]?.shapes.state ?? [];
  const isScalar = variant === 'scalar';

  return (
    <Reveal index={index} className="intro-card-stat" >
      <div className="intro-card-stat__head" style={{ borderTopColor: colour }}>
        <strong>{isScalar ? <Term term="scalar-decay" /> : <Term term="channel-wise-decay" />}</strong>
      </div>
      <dl className="intro-card-stat__rows">
        <div>
          <dt>
            <Term term="decay-gate">decay gate α</Term>, per layer
          </dt>
          <dd>
            <code>{shape(layer.alpha_shape)}</code>
          </dd>
        </div>
        <div>
          <dt>Numbers per token</dt>
          <dd>
            <CountUp to={perToken} />
          </dd>
        </div>
        <div>
          <dt>Decay projection</dt>
          <dd>
            <code>{isScalar ? '[1 × 128]' : '[128 × 128]'}</code>
          </dd>
        </div>
        <div>
          <dt>Memory grid per layer</dt>
          <dd>
            <code>{memory.length === 2 ? `${memory[0]} × ${memory[1]}` : '—'}</code>
          </dd>
        </div>
        <div>
          <dt>
            <Term term="parameter">Parameters</Term> in the whole model
          </dt>
          <dd className="intro-card-stat__big" style={{ color: colour }}>
            <CountUp to={gate.parameters} format={intFmt} />
          </dd>
        </div>
      </dl>
    </Reveal>
  );
}

/** Results: the two decay designs side by side, with real tensor shapes and parameter counts from the export. */
export function DimensionCards({ api }: { api: ApiResult<LayoutFixSummary> }) {
  if (api.loading) return <Skeleton lines={5} label="Loading saved model shapes" />;
  const summary = api.data;
  if (!summary) return <p className="intro-note">The model shapes come from the examiner server. Start it to see them.</p>;

  const scalarParams = summary.gates.scalar.parameters;
  const channelParams = summary.gates['channel-wise'].parameters;
  const extra = channelParams - scalarParams;
  const extraPct = (extra / scalarParams) * 100;

  return (
    <div className="intro-dims">
      <div className="intro-dims__cards">
        <GranularityCard variant="scalar" colour={SCALAR} summary={summary} index={0} />
        <GranularityCard variant="channel-wise" colour={CHANNEL} summary={summary} index={1} />
      </div>

      <Reveal index={2}>
        <h4 className="intro-subhead">Total parameters</h4>
        <AnimatedBars
          data={[
            { label: 'Scalar decay', value: scalarParams, color: SCALAR },
            { label: 'Channel-wise decay', value: channelParams, color: CHANNEL },
          ]}
          max={channelParams}
          format={intFmt}
        />
        <p className="intro-dims__delta">
          Channel-wise adds <strong><CountUp to={extra} format={intFmt} /></strong> parameters, an increase of{' '}
          <strong><CountUp to={extraPct} format={(n) => `${n.toFixed(2)}%`} /></strong>.
        </p>
        <WhatThisShows>
          the memory grid is identical in both designs. Only the decay gate differs, and it adds the parameters shown above.
        </WhatThisShows>
      </Reveal>
    </div>
  );
}

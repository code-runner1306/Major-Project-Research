import { useEffect, useState } from 'react';
import { useReducedMotion } from 'motion/react';
import type { LayoutFixSummary } from '../../api';
import { Heatmap } from '../../components/Heatmap';
import { TokenStrip } from '../../components/TokenStrip';
import { WhatThisShows } from '../../components/WhatThisShows';
import type { ApiResult } from '../../components/useApi';
import { CountUp, Skeleton } from './primitives';

const TRACE_SCALAR = 'A4_distance_and_rewrites/scalar';
const TRACE_CHANNEL = 'A4_distance_and_rewrites/channel-wise';
const STAGES = ['Before the write', 'Faded', 'After the write'] as const;
type Stage = 0 | 1 | 2;

const transpose = (rows: number[][]): number[][] => {
  const width = rows[0]?.length ?? 0;
  return Array.from({ length: width }, (_, c) => rows.map((row) => row[c]));
};

const absMax = (mats: number[][][]) => {
  let m = 0;
  for (const mat of mats) for (const row of mat) for (const v of row) m = Math.max(m, Math.abs(v));
  return m || 1;
};

/**
 * Real checkpoint data. Top: decay gate α over the 64 tokens of the showcase sequence, one row for scalar decay and one row
 * per memory column for channel-wise decay. Bottom: one memory step from the trained model (A4 arm, layer 4), in three stages.
 */
export function RealMemory({ api }: { api: ApiResult<LayoutFixSummary> }) {
  const reduce = useReducedMotion() ?? false;
  const [layer, setLayer] = useState(3);
  const [stage, setStage] = useState<Stage>(0);
  const [auto, setAuto] = useState(true);

  // Cycle the three stages while the user has not picked one; off with reduced motion.
  useEffect(() => {
    if (!auto || reduce) return;
    const t = window.setInterval(() => setStage((s) => ((s + 1) % 3) as Stage), 1600);
    return () => window.clearInterval(t);
  }, [auto, reduce]);

  if (api.loading) return <Skeleton lines={6} label="Loading real checkpoint traces" />;
  const s = api.data;
  if (!s) return <p className="intro-note">The real traces come from the examiner server. Start it to see them.</p>;

  const scalarAlpha = s.gates.scalar.layers[layer].alpha;
  const channelAlpha = s.gates['channel-wise'].layers[layer].alpha;
  const scalarRows = [scalarAlpha.map((row) => row[0])];
  const channelRows = transpose(channelAlpha);

  const tS = s.delta_rule[TRACE_SCALAR];
  const tC = s.delta_rule[TRACE_CHANNEL];
  const stagesOf = (t: typeof tS) => [t.state_before, t.state_decayed, t.state_after];
  const scalarStages = stagesOf(tS);
  const channelStages = stagesOf(tC);
  const scalarMax = absMax(scalarStages);
  const channelMax = absMax(channelStages);
  const worstGap = Math.max(...Object.values(s.delta_rule).map((t) => t.max_abs_error_vs_model_recurrence));

  return (
    <div className="intro-inside">
      <h4 className="intro-subhead">Decay gate over the showcase sequence (64 tokens)</h4>
      <div className="intro-seg" role="radiogroup" aria-label="Layer">
        {[0, 1, 2, 3].map((i) => (
          <button
            key={i}
            type="button"
            role="radio"
            aria-checked={layer === i}
            className={`intro-seg__btn${layer === i ? ' is-on' : ''}`}
            onClick={() => setLayer(i)}
          >
            Layer {i + 1}
          </button>
        ))}
      </div>
      <div className="intro-inside__alpha">
        <div>
          <span className="intro-inside__label">Scalar decay: 1 row</span>
          <div className="intro-strip">
            <Heatmap matrix={scalarRows} cellSize={4} mode="sequential" max={1} title="α, scalar" />
          </div>
        </div>
        <div>
          <span className="intro-inside__label">Channel-wise decay: {channelRows.length} rows, one per column</span>
          <Heatmap matrix={channelRows} cellSize={4} mode="sequential" max={1} title="α, channel-wise" />
        </div>
      </div>
      <WhatThisShows>
        darker means the memory is forgotten faster. Scalar decay is one number per token; channel-wise decay gives every
        column its own number, so the rows differ.
      </WhatThisShows>

      <div className="intro-strip intro-strip--tokens">
        <TokenStrip tokens={s.showcase.tokens} roles={s.showcase.roles} compactFiller />
      </div>

      <h4 className="intro-subhead">One real memory step, layer 4 (A4 arm)</h4>
      <div className="intro-seg" role="radiogroup" aria-label="Stage">
        {STAGES.map((label, i) => (
          <button
            key={label}
            type="button"
            role="radio"
            aria-checked={stage === i}
            className={`intro-seg__btn${stage === i ? ' is-on' : ''}`}
            onClick={() => {
              setAuto(false);
              setStage(i as Stage);
            }}
          >
            {label}
          </button>
        ))}
      </div>
      <div className="intro-inside__memory">
        <div>
          <Heatmap matrix={scalarStages[stage]} cellSize={7} mode="diverging" max={scalarMax} title="Scalar decay, 32 × 32 crop" />
          <p className="intro-inside__stat">
            β = {tS.beta.toFixed(4)} · read error{' '}
            <CountUp to={tS.read_error_before} format={(n) => n.toFixed(4)} /> → <CountUp to={tS.read_error_after} format={(n) => n.toFixed(4)} />
          </p>
        </div>
        <div>
          <Heatmap matrix={channelStages[stage]} cellSize={7} mode="diverging" max={channelMax} title="Channel-wise decay, 32 × 32 crop" />
          <p className="intro-inside__stat">
            β = {tC.beta.toFixed(4)} · read error{' '}
            <CountUp to={tC.read_error_before} format={(n) => n.toFixed(4)} /> → <CountUp to={tC.read_error_after} format={(n) => n.toFixed(4)} />
          </p>
        </div>
      </div>
      <WhatThisShows>
        the top-left 32 × 32 corner of the layer-4 memory at token {tS.token}, before the write, after fading, and after the
        write. Red is positive and blue is negative; each panel uses its own scale.
      </WhatThisShows>
      <p className="intro-inside__check">
        Our hand calculation of this step matches the model&apos;s own recurrence. The largest difference across the four traces
        is <strong>{worstGap.toExponential(1)}</strong>, which is floating-point rounding.
      </p>
    </div>
  );
}

import type { ReactNode } from 'react';
import { AnimatedLines } from '../../components/AnimatedLines';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote } from './common';
import { useLengthRun } from './data';
import { modelLabel, type ModelRef, type RunResponse } from './types';

const LAYER_COLORS = ['var(--tok-value)', 'var(--accent)', 'var(--c-arm-a0)', 'var(--text)'];

/** Decay-only survival: for each layer, the product of decay gates over the first n filler tokens (median). */
export function SurvivalChart({ run }: { run: RunResponse }) {
  if (run.filler === 0) {
    return <p className="ln-note">No filler is inserted, so nothing decays. Choose a filler count to see the curve.</p>;
  }
  const series = run.survival.layers.map((layer, i) => ({
    name: `layer ${layer.layer}`,
    color: LAYER_COLORS[i % LAYER_COLORS.length],
    ys: layer.log10_p50.map((v) => Math.pow(10, v)),
  }));
  return (
    <AnimatedLines
      series={series}
      xs={run.survival.steps}
      log
      yMax={1}
      xLabel="filler tokens inserted"
      yLabel="product of decay gates (log)"
    />
  );
}

/**
 * Live run of one checkpoint: the survival chart beside a side column holding the controls, the accuracy line
 * and, when a saved diagnostic exists for this model and filler count, a check against it.
 */
export function SurvivalPanel({
  model,
  filler,
  examples,
  controls,
}: {
  model: ModelRef;
  filler: number;
  examples: number;
  controls?: ReactNode;
}) {
  const state = useLengthRun({ ...model, filler, examples });
  const run = state.data;
  return (
    <div className="ln-split">
      <div className="ln-split__main">
        {run ? (
          <SurvivalChart run={run} />
        ) : (
          <Remote state={state} what="the live run" label="running the checkpoint on this CPU" tall>
            {() => null}
          </Remote>
        )}
        <WhatThisShows>Median product of decay gates over the filler, one line per layer.</WhatThisShows>
      </div>
      <aside className="ln-split__aside">
        {controls}
        {run && (
          <p className="ln-note">
            {modelLabel(model)}: {run.correct} of {run.total} answers correct at {run.length.toLocaleString()} tokens.
          </p>
        )}
        {run?.saved && (
          <table className="ln-table" aria-label="Live against saved diagnostic">
            <thead>
              <tr>
                <th>layer</th>
                <th>live</th>
                <th>saved</th>
              </tr>
            </thead>
            <tbody>
              {run.at_filler.map((row) => {
                const saved = run.saved?.layers.find((l) => l.layer === row.layer);
                return (
                  <tr key={row.layer}>
                    <td>{row.layer}</td>
                    <td>{row.log10_p50.toFixed(2)}</td>
                    <td>{saved ? saved.log10_product_p50.toFixed(2) : '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </aside>
    </div>
  );
}

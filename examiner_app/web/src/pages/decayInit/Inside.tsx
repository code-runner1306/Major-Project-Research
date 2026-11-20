import { useMemo, useState } from 'react';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, Segmented, conditionLabel, halfLifeTokens } from './common';
import { useDecayGates } from './data';
import { MiniLines, type MiniSeries } from './MiniLines';
import type { GateModel, Gates } from './types';

const LAYERS = [1, 2, 3, 4] as const;
const CHANNELS = Array.from({ length: 128 }, (_, i) => i + 1);

const keyOf = (m: GateModel) => `${m.init}/${m.variant}/${m.seed}`;

function sorted(values: number[]): number[] {
  return [...values].sort((a, b) => a - b);
}

/** Native select over the 12 saved models. */
function ModelPicker({ models, value, onChange }: { models: GateModel[]; value: GateModel; onChange: (m: GateModel) => void }) {
  return (
    <label className="di-field di-field--row">
      <span className="di-field__label">Model</span>
      <select
        value={keyOf(value)}
        onChange={(e) => {
          const next = models.find((m) => keyOf(m) === e.target.value);
          if (next) onChange(next);
        }}
      >
        {models.map((m) => (
          <option key={keyOf(m)} value={keyOf(m)}>
            {conditionLabel(m.init, m.variant)}, seed {m.seed}
          </option>
        ))}
      </select>
    </label>
  );
}

/** Left: average half-life per layer, at the start and after training. Right: channels of one layer, sorted. */
function InsideView({ model }: { model: GateModel }) {
  const [layer, setLayer] = useState<number>(4);
  const hasStart = model.layers[0].start !== null;

  const perLayer = useMemo<MiniSeries[]>(() => {
    const after: MiniSeries = {
      name: 'after training',
      ys: model.layers.map((l) => halfLifeTokens(l.trained.mean)),
      color: 'var(--c-channel)',
    };
    if (!hasStart) return [after];
    const start: MiniSeries = {
      name: 'start (recorded)',
      ys: model.layers.map((l) => halfLifeTokens(l.start!.mean)),
      color: 'var(--c-scalar)',
      dashed: true,
    };
    return [start, after];
  }, [model, hasStart]);

  const selected = model.layers[layer - 1];
  const channelSeries = useMemo<MiniSeries[]>(() => {
    const after: MiniSeries = {
      name: 'after training',
      ys: sorted(selected.trained.per_channel.map((m) => halfLifeTokens(m))),
      color: 'var(--c-channel)',
    };
    if (!selected.start) return [after];
    const start: MiniSeries = {
      name: 'start (recorded)',
      ys: sorted(selected.start.per_channel.map((m) => halfLifeTokens(m))),
      color: 'var(--c-scalar)',
      dashed: true,
    };
    return [start, after];
  }, [selected]);

  return (
    <Panels
      label="Inside"
      items={[
        {
          label: 'Average by layer',
          content: (
            <div className="di-inside__panel">
              <MiniLines series={perLayer} xs={[...LAYERS]} yLabel="half-life" xLabel="layer" log />
              {!hasStart && <p className="di-note">The default start is not recorded, so only the trained values are shown.</p>}
            </div>
          ),
        },
        {
          label: 'Channels in one layer',
          content:
            model.variant === 'channel-wise' ? (
              <div className="di-inside__panel">
                <Segmented label="layer" options={LAYERS.map((n) => ({ id: n, label: `layer ${n}` }))} value={layer} onChange={setLayer} />
                <MiniLines series={channelSeries} xs={CHANNELS} yLabel="half-life" xLabel="channel, sorted" log />
              </div>
            ) : (
              <p className="di-note">Scalar decay has one value per layer, so there are no channels to compare.</p>
            ),
        },
      ]}
    />
  );
}

function InsideBody({ gates }: { gates: Gates }) {
  const initial =
    gates.models.find((m) => m.init === 'mean-decay' && m.variant === 'channel-wise' && m.seed === 42) ?? gates.models[0];
  const [model, setModel] = useState<GateModel>(initial);
  return (
    <div className="di-inside">
      <ModelPicker models={gates.models} value={model} onChange={setModel} />
      <InsideView model={model} />
      <WhatThisShows>Half-life is the average decay in tokens: a diagnostic, not a measured memory lifetime.</WhatThisShows>
    </div>
  );
}

export function InsideVisual() {
  const gates = useDecayGates();
  return (
    <Remote state={gates} what="the saved gate values" tall>
      {(g) => <InsideBody gates={g} />}
    </Remote>
  );
}

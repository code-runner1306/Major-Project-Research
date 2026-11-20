import { useMemo, useState } from 'react';
import { TokenStrip } from '../../components/TokenStrip';
import { Unavailable } from '../../components/Unavailable';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, Segmented, conditionLabel, halfLifeTokens } from './common';
import { useLocalDecayModels, useTry } from './data';
import { MiniLines, type MiniSeries } from './MiniLines';
import { LAYOUTS, WHICH_WRITE_TEXT, layoutsFor, type LocalModel, type TryRequest, type TryResponse } from './types';

const REWRITES = [1, 2, 3, 4] as const;
const LAYERS = [1, 2, 3, 4] as const;

const keyOf = (m: LocalModel) => `${m.init}/${m.variant}/${m.seed}`;

function layoutName(id: string): string {
  return LAYOUTS.find((l) => l.id === id)?.label ?? id;
}

/** Plain-English reading of the answers, one line per question. */
function Answers({ data }: { data: TryResponse }) {
  return (
    <ul className="di-answers">
      {data.example.queries.map((q) => {
        const correct = q.prediction === q.gold;
        const question = q.kind === 'latest' ? 'newest value' : 'unrelated key';
        return (
          <li key={q.position} className={correct ? 'is-correct' : 'is-wrong'}>
            <span>
              {question}: answered token {q.prediction}, correct token {q.gold}.
            </span>{' '}
            <strong>{correct ? 'Correct.' : 'Wrong.'}</strong>
            {q.kind === 'latest' && (
              <span className="di-answers__source">
                {' '}The answer matches {WHICH_WRITE_TEXT[q.which_write] ?? q.which_write}.
              </span>
            )}
          </li>
        );
      })}
    </ul>
  );
}

/** Half-life per token for one layer: channel average, plus the fastest and slowest channel when channel-wise. */
function DecayLines({ data, layer }: { data: TryResponse; layer: number }) {
  const series = useMemo<MiniSeries[]>(() => {
    const alpha = data.alpha.layers[layer - 1].alpha;
    const isChannel = alpha[0].length > 1;
    const xs = alpha.map((_, t) => t);
    const lines: MiniSeries[] = [];
    if (isChannel) {
      lines.push({ name: 'slowest channel', ys: alpha.map((row) => halfLifeTokens(Math.max(...row))), color: 'var(--c-channel)' });
      lines.push({ name: 'fastest channel', ys: alpha.map((row) => halfLifeTokens(Math.min(...row))), color: 'var(--c-channel)', dashed: true });
    }
    const mean = (row: number[]) => row.reduce((a, b) => a + b, 0) / row.length;
    lines.push({ name: 'average', ys: alpha.map((row) => halfLifeTokens(mean(row))), color: 'var(--text)' });
    const start = data.start_mean?.[layer - 1];
    if (start !== undefined) {
      lines.push({ name: 'start average (recorded)', ys: xs.map(() => halfLifeTokens(start)), color: 'var(--c-scalar)', dashed: true });
    }
    return lines;
  }, [data, layer]);
  return <MiniLines series={series} xs={data.example.tokens.map((_, t) => t)} yLabel="half-life" xLabel="token" height={100} log />;
}

function TryResult({ data }: { data: TryResponse }) {
  const [layer, setLayer] = useState<number>(4);
  return (
    <div className="di-try__result">
      <TokenStrip tokens={data.example.tokens} roles={data.example.roles} compactFiller />
      <Answers data={data} />
      <div className="di-try__decay">
        <Segmented label="layer" options={LAYERS.map((n) => ({ id: n, label: `layer ${n}` }))} value={layer} onChange={setLayer} />
        <DecayLines data={data} layer={layer} />
      </div>
      <WhatThisShows>Each token’s decay in one layer, as a half-life in tokens.</WhatThisShows>
    </div>
  );
}

function TryControls({ models }: { models: LocalModel[] }) {
  const [model, setModel] = useState<LocalModel>(
    () => models.find((m) => m.init === 'mean-decay' && m.variant === 'channel-wise' && m.seed === 42) ?? models[0],
  );
  const [layout, setLayout] = useState<string>('block_first');
  const [rewrites, setRewrites] = useState<number>(3);
  const [distance, setDistance] = useState(8);
  const [exampleSeed, setExampleSeed] = useState(7);

  const effectiveLayout = layoutsFor(rewrites).includes(layout) ? layout : 'block_first';
  const body: TryRequest = {
    init: model.init,
    variant: model.variant,
    seed: model.seed,
    layout: effectiveLayout,
    rewrites,
    distance,
    example_seed: exampleSeed,
  };
  const live = useTry(body);

  return (
    <div className="di-try">
      <div className="di-controls">
        <label className="di-field">
          <span className="di-field__label">Model</span>
          <select
            value={keyOf(model)}
            onChange={(e) => {
              const next = models.find((m) => keyOf(m) === e.target.value);
              if (next) setModel(next);
            }}
          >
            {models.map((m) => (
              <option key={keyOf(m)} value={keyOf(m)}>
                {conditionLabel(m.init, m.variant)}, seed {m.seed}
              </option>
            ))}
          </select>
        </label>
        <label className="di-field">
          <span className="di-field__label">Layout</span>
          <select value={effectiveLayout} onChange={(e) => setLayout(e.target.value)}>
            {LAYOUTS.filter((l) => layoutsFor(rewrites).includes(l.id)).map((l) => (
              <option key={l.id} value={l.id}>
                {layoutName(l.id)}
              </option>
            ))}
          </select>
        </label>
        <div className="di-field">
          <span className="di-field__label">Rewrites</span>
          <Segmented options={REWRITES.map((n) => ({ id: n, label: String(n) }))} value={rewrites} onChange={setRewrites} label="rewrites" />
        </div>
        <label className="di-field">
          <span className="di-field__label">Distance: {distance} tokens</span>
          <input type="range" min={8} max={24} value={distance} onChange={(e) => setDistance(Number(e.target.value))} />
        </label>
        <div className="di-field">
          <span className="di-field__label">Seed</span>
          <div className="di-inline">
            <input
              type="number"
              min={0}
              value={exampleSeed}
              onChange={(e) => setExampleSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
            />
            <button type="button" className="di-btn" onClick={() => setExampleSeed(Math.floor(Math.random() * 10000))}>
              New seed
            </button>
          </div>
        </div>
      </div>
      <Remote state={live} what="this example" tall>
        {(data) => <TryResult data={data} />}
      </Remote>
    </div>
  );
}

export function TryVisual() {
  const models = useLocalDecayModels();
  return (
    <Remote state={models} what="the local models" tall>
      {(list) =>
        list.length === 0 ? (
          <Unavailable reason="No local overwrite checkpoints were found on this machine." />
        ) : (
          <TryControls models={list} />
        )
      }
    </Remote>
  );
}

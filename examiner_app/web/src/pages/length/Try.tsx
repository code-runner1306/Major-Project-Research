import { useState } from 'react';
import { Panels } from '../../components/Panels';
import { Stat } from '../../components/Stat';
import { WhatThisShows } from '../../components/WhatThisShows';
import { ModelSelect, Remote } from './common';
import { useLengthRun } from './data';
import { SurvivalChart } from './Survival';
import { MAX_FILLER, pct, type LocalModel, type ModelRef, type RunResponse } from './types';

const PRESETS = [0, 192, 480, 960];
const EXAMPLE_COUNTS = [1, 4, 8, 16];

/** Answers for the first example, with the accuracy over all examples. */
function AnswersView({ run }: { run: RunResponse }) {
  return (
    <div className="ln-try__answers">
      <div className="ln-stats">
        <div className="ln-stat-card">
          <Stat value={run.accuracy} format={(v) => pct(v)} label={`correct over ${run.examples} examples`} />
        </div>
        <div className="ln-stat-card">
          <Stat value={run.correct} format={(v) => `${Math.round(v)} / ${run.total}`} label="answers correct" />
        </div>
      </div>
      <div className="ln-blocks" aria-label="Input of the first example">
        <div className="ln-seg is-write" style={{ flex: '0 0 96px' }}>8 write tokens</div>
        <div className="ln-seg is-filler" style={{ flex: Math.max(run.filler, 1) }}>{run.filler} filler</div>
        <div className="ln-seg is-rest" style={{ flex: 56 }}>56 query keys and filler</div>
      </div>
      <table className="ln-table" aria-label="Answers for the first example">
        <thead>
          <tr>
            <th>position</th>
            <th>expected</th>
            <th>model</th>
            <th>result</th>
          </tr>
        </thead>
        <tbody>
          {run.example.queries.map((q) => (
            <tr key={q.position} className={q.correct ? '' : 'is-wrong'}>
              <td>{q.position}</td>
              <td>{q.gold}</td>
              <td>{q.prediction}</td>
              <td>{q.correct ? 'correct' : 'wrong'}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

/** Try-it: choose a checkpoint, a filler count and a number of examples, then see the answers and the decay curve. */
export function LengthTryVisual({ models }: { models: LocalModel[] }) {
  const [model, setModel] = useState<ModelRef>(() => models[0] ?? { init: 'default', variant: 'scalar', seed: 42 });
  const [filler, setFiller] = useState(192);
  const [examples, setExamples] = useState(8);
  const state = useLengthRun({ ...model, filler, examples });

  return (
    <div className="ln-try">
      <div className="ln-try__controls">
        <ModelSelect models={models} value={model} onChange={setModel} />
        <label className="ln-field">
          <span className="ln-field__label">Filler tokens (0 to {MAX_FILLER})</span>
          <input
            type="number"
            min={0}
            max={MAX_FILLER}
            value={filler}
            onChange={(e) => {
              const v = Math.round(Number(e.target.value));
              if (Number.isFinite(v)) setFiller(Math.min(MAX_FILLER, Math.max(0, v)));
            }}
          />
        </label>
        <div className="ln-presets" role="group" aria-label="Filler presets">
          {PRESETS.map((p) => (
            <button key={p} type="button" aria-pressed={filler === p} onClick={() => setFiller(p)}>
              {p}
            </button>
          ))}
        </div>
        <label className="ln-field">
          <span className="ln-field__label">Examples</span>
          <select value={examples} onChange={(e) => setExamples(Number(e.target.value))}>
            {EXAMPLE_COUNTS.map((n) => (
              <option key={n} value={n}>
                {n}
              </option>
            ))}
          </select>
        </label>
      </div>
      <div className="ln-try__output">
        <Remote state={state} what="the live run" label="running the checkpoint on this CPU" tall>
          {(run) => (
            <Panels
              label="Try results"
              items={[
                { label: 'Answers', content: <AnswersView run={run} /> },
                {
                  label: 'Decay curve',
                  content: (
                    <div className="ln-try__chart">
                      <SurvivalChart run={run} />
                      <WhatThisShows>Decay-only product over the filler; it omits write strength.</WhatThisShows>
                    </div>
                  ),
                },
              ]}
            />
          )}
        </Remote>
      </div>
    </div>
  );
}

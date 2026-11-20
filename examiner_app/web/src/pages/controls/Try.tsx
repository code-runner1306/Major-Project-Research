import { useState } from 'react';
import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { TokenStrip } from '../../components/TokenStrip';
import { Unavailable } from '../../components/Unavailable';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, Segmented, pct } from './common';
import {
  modelLabel,
  useZoologyTry,
  type ControlsSummary,
  type TryQuery,
  type ZoologyEvaluateRequest,
  type ZoologyEvaluateResponse,
  type ZoologyModelRow,
} from './data';

const EXAMPLE_COUNTS = [16, 32, 64] as const;
const SOURCE_LABEL: Record<string, string> = {
  correct: 'the correct value',
  position_anchored: 'anchored token',
  other_pair: "another key's value",
  other: 'some other token',
};
const OUTCOME_TEXT: Record<TryQuery['outcome'], string> = {
  correct: 'correct',
  position_anchored: 'position-anchored token',
  other_pair: "another key's value",
  other: 'other token',
};

/** Section 6: the live Zoology try-it, then KATA (results only) and a browser for the saved results. */
export function TryVisual({ summary }: { summary: ControlsSummary }) {
  return (
    <Panels
      label="Try views"
      items={[
        { label: 'Try a Zoology model (live)', content: <TryZoology summary={summary} /> },
        { label: 'KATA models', content: <KataPanel summary={summary} /> },
        { label: 'Saved results', content: <SavedResults summary={summary} /> },
      ]}
    />
  );
}

function TryZoology({ summary }: { summary: ControlsSummary }) {
  const models = summary.zoology.models;
  const local = models.filter((m) => m.local);
  const [choice, setChoice] = useState<number>(() => {
    const first = local.find((m) => m.init === 'default' && m.variant === 'scalar' && m.seed === 44) ?? local[0];
    return first ? models.indexOf(first) : 0;
  });
  const [layout, setLayout] = useState<string>('shift_1');
  const [examples, setExamples] = useState<number>(32);
  const [exampleSeed, setExampleSeed] = useState<number>(44);

  const model: ZoologyModelRow | undefined = models[choice];
  const body: ZoologyEvaluateRequest | null =
    model && model.local
      ? { init: model.init, variant: model.variant, seed: model.seed, layout, examples, example_seed: exampleSeed, show_examples: 2 }
      : null;
  const ev = useZoologyTry(body);

  if (local.length === 0) {
    return <Unavailable reason="None of the Zoology checkpoints are on this machine." />;
  }

  return (
    <div className="ctl-try">
      <div className="ctl-controls">
        <label className="ctl-field">
          <span className="ctl-field__label">Model</span>
          <select value={choice} onChange={(e) => setChoice(Number(e.target.value))}>
            {models.map((m, i) => (
              <option key={`${m.init}-${m.variant}-${m.seed}`} value={i} disabled={!m.local}>
                {modelLabel(m)}
                {m.local ? '' : ' (unavailable)'}
              </option>
            ))}
          </select>
        </label>
        <label className="ctl-field">
          <span className="ctl-field__label">Layout</span>
          <select value={layout} onChange={(e) => setLayout(e.target.value)}>
            {summary.zoology.layouts.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
        <div className="ctl-field">
          <span className="ctl-field__label">Questions</span>
          <Segmented options={EXAMPLE_COUNTS.map((n) => ({ id: n, label: String(n) }))} value={examples} onChange={setExamples} label="examples" />
        </div>
        <label className="ctl-field">
          <span className="ctl-field__label">Example seed</span>
          <input type="number" min={0} value={exampleSeed} onChange={(e) => setExampleSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
        </label>
      </div>
      {body === null ? (
        <Unavailable reason="This Zoology model's checkpoint is not on this machine." />
      ) : (
        <Remote state={ev} what="this Zoology model">
          {(r) => <TryBody r={r} />}
        </Remote>
      )}
    </div>
  );
}

function TryBody({ r }: { r: ZoologyEvaluateResponse }) {
  const sources: BarDatum[] = ['correct', 'position_anchored', 'other_pair', 'other'].map((k) => ({
    label: SOURCE_LABEL[k],
    value: r.answer_sources[k] ?? 0,
    color: k === 'correct' ? undefined : 'var(--accent)',
  }));
  const byQuery: BarDatum[] = r.accuracy_by_query.map((v, i) => ({ label: `question ${i + 1}`, value: v ?? 0 }));
  const first = r.shown[0];
  return (
    <div className="ctl-try__body">
      <p className="ctl-try__summary">
        <strong>{r.layout_label}:</strong> {pct(r.accuracy, 1)} of {r.n_answers} answers are the correct value. {r.layout_description}
      </p>
      <Panels
        label="Try readouts"
        items={[
          {
            label: 'Where answers come from',
            content: (
              <div className="ctl-try__panel">
                <AnimatedBars data={sources} max={1} format={(v) => pct(v, 1)} />
                <WhatThisShows>Share of answers by source. Anchored token: where the value would sit in the standard layout.</WhatThisShows>
              </div>
            ),
          },
          {
            label: 'Accuracy by question',
            content: (
              <div className="ctl-try__panel">
                <AnimatedBars data={byQuery} max={1} format={(v) => pct(v, 0)} />
                <WhatThisShows>Questions in the order they are asked within each example.</WhatThisShows>
              </div>
            ),
          },
          {
            label: 'One example',
            content: first ? (
              <div className="ctl-try__panel">
                <TokenStrip tokens={first.tokens} roles={first.roles} compactFiller />
                <ul className="ctl-try__queries">
                  {first.queries.map((q) => (
                    <li key={q.position} className={q.outcome === 'correct' ? 'is-ok' : 'is-bad'}>
                      Question at token {q.position}: answered {q.prediction}, the value was {q.gold}. {OUTCOME_TEXT[q.outcome]}
                      {q.picked_slot !== null && ` (table slot ${q.picked_slot})`}
                    </li>
                  ))}
                </ul>
                <WhatThisShows>Each square is one token; the answer comes from the token the model picked.</WhatThisShows>
              </div>
            ) : (
              <p className="ctl-muted">No example to show.</p>
            ),
          },
        ]}
      />
    </div>
  );
}

const PASS_TEXT: Record<string, string> = {
  one_pass: 'one write pass',
  two_pass: 'two write passes',
  three_pass: 'three write passes',
  four_pass: 'four write passes',
  partial_third: 'third pass on half the keys',
};

/** 'three_pass_K32' -> 'three write passes, 32 keys'. Plain words only; the ids stay in the data. */
function cellLabel(id: string): string {
  const [pass, keys] = id.split('_K');
  const text = PASS_TEXT[pass] ?? pass;
  return keys ? `${text}, ${keys} keys` : text;
}

function KataPanel({ summary }: { summary: ControlsSummary }) {
  const k = summary.kata;
  return (
    <div className="ctl-try__panel">
      <Unavailable reason="The KATA models are not on this machine. Only logs and saved results were kept, so nothing can be run here." />
      <ul className="ctl-try__facts">
        <li>{k.readings.replication}</li>
        <li>{k.readings.shortcut}</li>
        <li>{k.readings.error_identity}</li>
      </ul>
      <WhatThisShows>Results only. The saved-results tab shows the full table.</WhatThisShows>
    </div>
  );
}

function SavedResults({ summary }: { summary: ControlsSummary }) {
  const z = summary.zoology;
  const k = summary.kata;
  const [layout, setLayout] = useState<string>(z.layouts[0]?.id ?? '');
  const [keys, setKeys] = useState<number>(32);
  const layoutInfo = z.layouts.find((l) => l.id === layout) ?? z.layouts[0];
  const keyCounts = [...new Set(k.cell_names.map((c) => Number(c.split('_K')[1])).filter((n) => !Number.isNaN(n)))].sort((a, b) => a - b);
  const testsForKeys = k.cell_names.filter((c) => c.endsWith(`_K${keys}`));

  return (
    <div className="ctl-saved">
      <div className="ctl-saved__block">
        <div className="ctl-controls">
          <label className="ctl-field">
            <span className="ctl-field__label">Zoology layout</span>
            <select value={layoutInfo?.id} onChange={(e) => setLayout(e.target.value)}>
              {z.layouts.map((l) => (
                <option key={l.id} value={l.id}>
                  {l.label}
                </option>
              ))}
            </select>
          </label>
        </div>
        <table className="ctl-table ctl-table--narrow">
          <thead>
            <tr>
              <th>Model</th>
              <th>Accuracy</th>
            </tr>
          </thead>
          <tbody>
            {z.models.map((m) => (
              <tr key={`${m.init}-${m.variant}-${m.seed}`} className={m.shortcut ? 'is-flag' : undefined}>
                <th scope="row">{modelLabel(m)}</th>
                <td>{pct(m.accuracy[layoutInfo?.id ?? ''] ?? 0, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="ctl-saved__block">
        <div className="ctl-controls">
          <label className="ctl-field">
            <span className="ctl-field__label">KATA keys in the sequence</span>
            <select value={keys} onChange={(e) => setKeys(Number(e.target.value))}>
              {keyCounts.map((n) => (
                <option key={n} value={n}>
                  {n}
                </option>
              ))}
            </select>
          </label>
        </div>
        <table className="ctl-table ctl-table--narrow">
          <thead>
            <tr>
              <th>Test</th>
              {k.learning_rates.map((lr) => (
                <th key={lr}>lr {lr.replace('lr_', '')}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {testsForKeys.map((c) => (
              <tr key={c}>
                <th scope="row">{cellLabel(c)}</th>
                {k.learning_rates.map((lr) => (
                  <td key={lr}>{k.cells[c][lr] === null ? '—' : pct(k.cells[c][lr] ?? 0, 1)}</td>
                ))}
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}

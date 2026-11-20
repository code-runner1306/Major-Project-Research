import { useEffect, useState, type ReactNode } from 'react';
import { AnimatedBars } from '../../components/AnimatedBars';
import { AnimatedLines } from '../../components/AnimatedLines';
import { Panels } from '../../components/Panels';
import { Stat } from '../../components/Stat';
import { WhatThisShows } from '../../components/WhatThisShows';
import { ApiFailure, ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import { QUERIES, pct, runTry, type Profile, type ProfileCell, type Query, type RunSummary, type Summary, type TryResult } from './data';

/** The shape useRetrying returns (see pages/dataset/shared.tsx). */
export interface RemoteState<T> {
  data: T | null;
  error: ApiFailure | null;
  waiting: boolean;
  pending: boolean;
}

/** Skeleton while loading, retry notice while the server starts, error panel on failure. */
export function Loaded<T>({ state, what, children }: { state: RemoteState<T>; what: string; children: (data: T) => ReactNode }) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={`loading ${what}`} tall />;
  return <div className={state.pending ? 'sr-pending' : undefined}>{children(state.data)}</div>;
}

const avg = (a: number, b: number) => (a + b) / 2;

export const QUERY_LABEL: Record<Query, string> = {
  target: 'Cleared key',
  survivor: 'Other workspace key',
  rewrite: 'Rewritten key',
  absent: 'Missing key',
};

/* ---------- Question: a whiteboard with two workspaces ---------- */

const RED_ROWS: [string, string][] = [
  ['key A', 'value 3'],
  ['key B', 'value 9'],
  ['key C', 'value 5'],
];
const BLUE_ROWS: [string, string][] = [
  ['key D', 'value 7'],
  ['key E', 'value 1'],
  ['key F', 'value 8'],
];

export function QuestionVisual() {
  const [cleared, setCleared] = useState(false);
  return (
    <div className="sr-question">
      <div className="sr-board" role="img" aria-label="Two workspaces, red and blue, each holding key-value pairs">
        <div className="sr-col">
          <div className="sr-col__name">Red workspace</div>
          {RED_ROWS.map(([k, v]) => (
            <div key={k} className={`sr-pair${cleared ? ' is-cleared' : ''}`}>
              <span>{k}</span>
              <span>{cleared ? 'empty' : v}</span>
            </div>
          ))}
        </div>
        <div className="sr-col sr-col--blue">
          <div className="sr-col__name">Blue workspace</div>
          {BLUE_ROWS.map(([k, v]) => (
            <div key={k} className="sr-pair">
              <span>{k}</span>
              <span>{v}</span>
            </div>
          ))}
        </div>
      </div>
      <div className="sr-controls">
        <button type="button" className="sr-btn" aria-pressed={cleared} onClick={() => setCleared((c) => !c)}>
          {cleared ? 'Undo the reset' : 'Send a reset to red'}
        </button>
      </div>
      <WhatThisShows>A reset should clear red and leave blue exactly as it was.</WhatThisShows>
    </div>
  );
}

/* ---------- Model: stock recurrent cells, two decay styles ---------- */

const PIPELINE = [
  { title: 'Tokens in', note: 'one token per step' },
  { title: 'Layer 1', note: 'memory table, updated each step' },
  { title: 'Layer 2', note: 'memory table, updated each step' },
  { title: 'Answer', note: 'one token out' },
];

function Pipeline() {
  return (
    <ol className="sr-pipeline" aria-label="Model pipeline">
      {PIPELINE.map((s, i) => (
        <li key={s.title} className="sr-pipeline__item">
          <div className="sr-pipeline__box">
            <strong>{s.title}</strong>
            <span>{s.note}</span>
          </div>
          {i < PIPELINE.length - 1 && (
            <span className="sr-pipeline__arrow" aria-hidden="true">
              →
            </span>
          )}
        </li>
      ))}
    </ol>
  );
}

function DecayStyles({ summary }: { summary: Summary }) {
  const params = (decay: string) => summary.runs.find((r) => r.decay === decay)?.parameters;
  const cards = [
    { name: 'Channel-wise decay', text: 'Each memory channel forgets at its own rate.', params: params('channel-wise') },
    { name: 'Scalar decay', text: 'One forgetting rate for the whole memory at each step.', params: params('scalar') },
  ];
  return (
    <div className="sr-decay-cards">
      {cards.map((c) => (
        <div key={c.name} className="sr-decay-card">
          <strong>{c.name}</strong>
          <p>{c.text}</p>
          <span className="sr-muted">{c.params === undefined ? '' : `${c.params.toLocaleString()} parameters`}</span>
        </div>
      ))}
    </div>
  );
}

export function ModelVisual({ state }: { state: RemoteState<Summary> }) {
  return (
    <>
      <Panels
        label="Model views"
        items={[
          { label: 'Pipeline', content: <Pipeline /> },
          { label: 'Decay styles', content: <Loaded state={state} what="the model sizes">{(s) => <DecayStyles summary={s} />}</Loaded> },
        ]}
      />
      <WhatThisShows>Both runs share one pipeline; only the decay style differs.</WhatThisShows>
    </>
  );
}

/* ---------- Results ---------- */

function PerQuery({ summary }: { summary: Summary }) {
  const q = summary.per_query;
  return (
    <AnimatedBars
      max={1}
      format={(v) => pct(v)}
      data={[
        { label: 'Cleared key, after reset', value: q.reset_target, note: 'empty is correct' },
        { label: 'Cleared key, no reset', value: q.noop_target, note: 'old value is correct' },
        { label: 'Other workspace key', value: avg(q.reset_survivor, q.noop_survivor), note: 'should keep its value' },
        { label: 'Rewritten key', value: avg(q.reset_rewrite, q.noop_rewrite), note: 'newest value is correct' },
        { label: 'Missing key', value: avg(q.reset_absent, q.noop_absent), note: 'empty is correct' },
      ]}
    />
  );
}

function ByLoad({ summary }: { summary: Summary }) {
  const rows = summary.by_load;
  return (
    <AnimatedLines
      xs={rows.map((r) => r.load)}
      yMax={1}
      yLabel="Share answered correctly"
      xLabel="Memory load (keys in the target workspace)"
      series={[
        { name: 'Cleared key, after reset', ys: rows.map((r) => r.reset_target), color: 'var(--accent)' },
        { name: 'Other workspace key', ys: rows.map((r) => avg(r.reset_survivor, r.noop_survivor)), color: 'var(--text)', dashed: true },
        { name: 'Rewritten key', ys: rows.map((r) => avg(r.reset_rewrite, r.noop_rewrite)), color: 'var(--c-channel)' },
      ]}
    />
  );
}

export function RunTable({ runs }: { runs: RunSummary[] }) {
  return (
    <table className="sr-table">
      <thead>
        <tr>
          <th>Decay</th>
          <th>Seed</th>
          <th>Cleared key after reset (lowest load)</th>
          <th>Other workspace key (highest)</th>
          <th>Both right (highest)</th>
        </tr>
      </thead>
      <tbody>
        {runs.map((run) => (
          <tr key={`${run.decay}-${run.seed}`}>
            <td>{run.decay}</td>
            <td>{run.seed}</td>
            <td>{pct(Math.min(...run.by_load.map((r) => r.reset_target)))}</td>
            <td>{pct(Math.max(...run.by_load.flatMap((r) => [r.reset_survivor, r.noop_survivor])), 2)}</td>
            <td>{pct(Math.max(...run.by_load.flatMap((r) => [r.reset_joint, r.noop_joint])), 2)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function ResultsVisual({ summary }: { summary: Summary }) {
  return (
    <div className="sr-results">
      <div className="sr-results__stat">
        <Stat value={summary.max_joint} format={(v) => pct(v)} label="best score with both keys right, any run or load" tone="bad" />
      </div>
      <Panels
        label="Results views"
        items={[
          { label: 'Each query', content: <PerQuery summary={summary} /> },
          { label: 'By memory load', content: <ByLoad summary={summary} /> },
          { label: 'Each run', content: <RunTable runs={summary.runs} /> },
        ]}
      />
      <WhatThisShows>
        Averages over {summary.runs.length} runs and {summary.task.memory_loads.length} memory loads, from the saved result files.
      </WhatThisShows>
    </div>
  );
}

/* ---------- Inside: what the saved checkpoint answers ---------- */

const KIND_COLOR: Record<string, string> = {
  correct: 'var(--c-good)',
  empty: 'var(--accent)',
  'old value of cleared key': 'var(--c-channel)',
  "other workspace's value": 'var(--c-scalar)',
  'something else': 'var(--border-strong)',
};

export function InsideVisual({
  state,
  loads,
  load,
  onLoad,
}: {
  state: RemoteState<Profile>;
  loads: number[];
  load: number;
  onLoad: (n: number) => void;
}) {
  return (
    <div className="sr-inside">
      <div className="sr-segmented" role="group" aria-label="Memory load">
        {loads.map((n) => (
          <button key={n} type="button" className="sr-btn" aria-pressed={load === n} onClick={() => onLoad(n)}>
            {n} {n === 1 ? 'key' : 'keys'}
          </button>
        ))}
      </div>
      <Loaded state={state} what="the checkpoint's answers">
        {(p) => {
          const cells = p.by_load[String(load)] ?? {};
          return (
            <>
              <ul className="sr-legend">
                {p.kinds.map((k) => (
                  <li key={k}>
                    <span className="sr-swatch" style={{ background: KIND_COLOR[k] }} />
                    {k}
                  </li>
                ))}
              </ul>
              <div className="sr-stack-rows">
                {(['noop', 'reset'] as const).flatMap((mode) =>
                  QUERIES.map((q) => (
                    <StackRow
                      key={`${mode}-${q}`}
                      label={`${mode === 'reset' ? 'After reset' : 'No reset'}: ${QUERY_LABEL[q]}`}
                      cell={cells[`${mode}_${q}`]}
                      kinds={p.kinds}
                    />
                  )),
                )}
              </div>
              <WhatThisShows>Each row splits one query's answers by what the saved checkpoint returned.</WhatThisShows>
            </>
          );
        }}
      </Loaded>
    </div>
  );
}

function StackRow({ label, cell, kinds }: { label: string; cell: ProfileCell | undefined; kinds: string[] }) {
  if (!cell) return null;
  const summary = kinds.map((k) => `${k} ${pct(cell[k] ?? 0, 0)}`).join(', ');
  return (
    <div className="sr-stack-row">
      <span className="sr-stack-row__label">{label}</span>
      <div className="sr-stack" role="img" aria-label={`${label}: ${summary}`}>
        {kinds.map((k) => {
          const share = cell[k] ?? 0;
          return share > 0 ? (
            <span key={k} className="sr-stack__seg" style={{ width: `${share * 100}%`, background: KIND_COLOR[k] }} title={`${k}: ${pct(share)}`} />
          ) : null;
        })}
      </div>
    </div>
  );
}

/* ---------- Try it: one fresh episode, one answer from the saved checkpoint ---------- */

export function TryVisual({ summary }: { summary: Summary }) {
  const [load, setLoad] = useState(2);
  const [reset, setReset] = useState(true);
  const [query, setQuery] = useState<Query>('target');
  const [seed, setSeed] = useState(7);
  const [result, setResult] = useState<TryResult | null>(null);
  const [error, setError] = useState<ApiFailure | null>(null);
  const [busy, setBusy] = useState(false);

  useEffect(() => {
    let cancelled = false;
    setBusy(true);
    runTry({ load, reset, query, seed })
      .then((r) => {
        if (cancelled) return;
        setResult(r);
        setError(null);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        setError(err instanceof ApiFailure ? err : new ApiFailure(String(err), null));
      })
      .finally(() => {
        if (!cancelled) setBusy(false);
      });
    return () => {
      cancelled = true;
    };
  }, [load, reset, query, seed]);

  return (
    <div className="sr-try">
      <div className="sr-try__controls">
        <div className="sr-segmented" role="group" aria-label="Memory load">
          {summary.task.memory_loads.map((n) => (
            <button key={n} type="button" className="sr-btn" aria-pressed={load === n} onClick={() => setLoad(n)}>
              {n} {n === 1 ? 'key' : 'keys'}
            </button>
          ))}
        </div>
        <div className="sr-segmented" role="group" aria-label="Reset">
          <button type="button" className="sr-btn" aria-pressed={reset} onClick={() => setReset(true)}>
            Reset the target workspace
          </button>
          <button type="button" className="sr-btn" aria-pressed={!reset} onClick={() => setReset(false)}>
            No reset
          </button>
        </div>
        <div className="sr-segmented" role="group" aria-label="Question">
          {QUERIES.map((q) => (
            <button key={q} type="button" className="sr-btn" aria-pressed={query === q} onClick={() => setQuery(q)}>
              {QUERY_LABEL[q]}
            </button>
          ))}
        </div>
        <label className="sr-seed">
          Example number{' '}
          <input
            type="number"
            min={0}
            max={99999}
            value={seed}
            onChange={(e) => setSeed(Math.max(0, Math.min(99999, Number(e.target.value) || 0)))}
          />
        </label>
      </div>
      {error && <ExampleError error={error} what="The checkpoint could not answer." />}
      {result && (
        <>
          <div className="sr-tokens" aria-label="The example, token by token">
            {result.tokens.map((t, i) => (
              <span
                key={i}
                className={`sr-tok sr-tok--${t.role}${i >= result.tokens.length - 2 ? ' is-question' : ''}`}
                title={t.label || t.role}
              >
                {t.role === 'filler' ? '' : t.label}
              </span>
            ))}
          </div>
          <div className="sr-answers">
            <div className="sr-answer">
              <span className="sr-muted">Correct answer</span>
              <strong>{result.expected_label}</strong>
            </div>
            <div className={`sr-answer ${result.correct ? 'is-right' : 'is-wrong'}`}>
              <span className="sr-muted">Saved model answers</span>
              <strong>{result.prediction_label}</strong>
              <span className="sr-badge">{result.correct ? 'correct' : 'wrong'}</span>
            </div>
          </div>
          <ul className="sr-top3">
            {result.top3.map((t) => (
              <li key={t.token}>
                <span>{t.label}</span>
                <span className="sr-top3__bar" style={{ width: `${Math.max(2, t.prob * 100)}%` }} />
                <span className="sr-muted">{pct(t.prob)}</span>
              </li>
            ))}
          </ul>
          <WhatThisShows>One fresh example from the pilot's generator, answered by the saved model.</WhatThisShows>
        </>
      )}
      {busy && !result && <Skeleton label="running the checkpoint" />}
    </div>
  );
}

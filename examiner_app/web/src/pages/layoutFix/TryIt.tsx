import { useEffect, useState } from 'react';
import {
  ARMS,
  ARM_CHANGE,
  ARM_LABEL,
  type Arm,
  type TryRequest,
  type TryResponse,
  type Variant,
} from '../../api';
import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { postJson } from '../../components/useApi';
import { Term } from '../../components/Term';
import { TokenStrip } from '../../components/TokenStrip';
import { WhatThisShows } from '../../components/WhatThisShows';
import { armColor, isBlockFirst, layoutsFor, TRY_LAYOUTS, whichWriteText } from './derive';
import { PlaybackControls, StatePlayer, usePlayback } from './StatePlayer';

type Seed = TryRequest['seed'];

const PRESETS: { label: string; arm: Arm; variant: Variant; seed: Seed; layout: TryRequest['layout']; rewrites: number }[] = [
  { label: 'A0 control, block-first', arm: 'A0_control', variant: 'scalar', seed: 42, layout: 'block_first', rewrites: 3 },
  { label: 'Same example on A4', arm: 'A4_distance_and_rewrites', variant: 'scalar', seed: 42, layout: 'block_first', rewrites: 3 },
];

/** Interactive try-it: pick an arm and an example, run it on the real checkpoint, and inspect the answers and internals. */
export function TryIt() {
  const [arm, setArm] = useState<Arm>('A0_control');
  const [variant, setVariant] = useState<Variant>('scalar');
  const [seed, setSeed] = useState<Seed>(42);
  const [layoutChoice, setLayoutChoice] = useState<TryRequest['layout']>('block_first');
  const [rewrites, setRewrites] = useState(3);
  const [distance, setDistance] = useState(8);
  const [exampleSeed, setExampleSeed] = useState(7);

  // A layout that does not exist for the chosen rewrite count falls back to block-first.
  const layout = layoutsFor(rewrites).includes(layoutChoice) ? layoutChoice : 'block_first';

  const req: TryRequest = { arm, variant, seed, layout, rewrites, distance, example_seed: exampleSeed };
  const key = JSON.stringify(req);

  const [result, setResult] = useState<{ key: string; data: TryResponse | null; error: string | null }>({
    key: '',
    data: null,
    error: null,
  });

  // Debounced: controls can change quickly, only the last settled request is sent.
  useEffect(() => {
    let cancelled = false;
    const body = JSON.parse(key) as TryRequest;
    const timer = window.setTimeout(() => {
      postJson<TryResponse>('/api/layout-fix/try', body)
        .then((data) => {
          if (!cancelled) setResult({ key, data, error: null });
        })
        .catch((err: unknown) => {
          if (!cancelled) setResult({ key, data: null, error: err instanceof Error ? err.message : String(err) });
        });
    }, 220);
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, [key]);

  const loading = result.key !== key;
  const data = result.data;
  // One playback for every memory view of the current example.
  const pb = usePlayback(data && !result.error ? data.example.tokens.length : 0);
  const skeleton = <div className="lf-skeleton" aria-label="loading example" />;

  return (
    <div className="lf-try">
      <div className="lf-controls">
        <div className="lf-presets" role="group" aria-label="presets">
          {PRESETS.map((p) => (
            <button
              key={p.label}
              type="button"
              className="lf-btn"
              onClick={() => {
                setArm(p.arm);
                setVariant(p.variant);
                setSeed(p.seed);
                setLayoutChoice(p.layout);
                setRewrites(p.rewrites);
              }}
            >
              {p.label}
            </button>
          ))}
        </div>

        <label className="lf-field">
          <span className="lf-field__label">Training data</span>
          <select value={arm} onChange={(e) => setArm(e.target.value as Arm)} title={ARM_CHANGE[arm]}>
            {ARMS.map((a) => (
              <option key={a} value={a}>
                {ARM_LABEL[a]}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="lf-field">
          <legend className="lf-field__label">
            <Term term="decay-granularity" />
          </legend>
          <div className="lf-segmented">
            {(['scalar', 'channel-wise'] as Variant[]).map((v) => (
              <button
                key={v}
                type="button"
                aria-pressed={variant === v}
                className={variant === v ? 'is-on' : ''}
                onClick={() => setVariant(v)}
              >
                {v}
              </button>
            ))}
          </div>
        </fieldset>

        <label className="lf-field">
          <span className="lf-field__label">
            <Term term="seed" /> (run)
          </span>
          <select value={seed} onChange={(e) => setSeed(Number(e.target.value) as Seed)}>
            {([42, 43, 44] as Seed[]).map((s) => (
              <option key={s} value={s}>
                seed {s}
              </option>
            ))}
          </select>
        </label>

        <label className="lf-field">
          <span className="lf-field__label">Example layout</span>
          <select value={layout} onChange={(e) => setLayoutChoice(e.target.value as TryRequest['layout'])}>
            {TRY_LAYOUTS.filter((l) => layoutsFor(rewrites).includes(l.id)).map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </label>

        <fieldset className="lf-field">
          <legend className="lf-field__label">Rewrites (1–4)</legend>
          <div className="lf-segmented">
            {[1, 2, 3, 4].map((n) => (
              <button key={n} type="button" aria-pressed={rewrites === n} className={rewrites === n ? 'is-on' : ''} onClick={() => setRewrites(n)}>
                {n}
              </button>
            ))}
          </div>
        </fieldset>

        <label className="lf-field">
          <span className="lf-field__label">Query distance: {distance} (8–24)</span>
          <input type="range" min={8} max={24} value={distance} onChange={(e) => setDistance(Number(e.target.value))} />
        </label>

        <div className="lf-field">
          <span className="lf-field__label">Example seed</span>
          <div className="lf-inline">
            <input
              type="number"
              min={0}
              value={exampleSeed}
              onChange={(e) => setExampleSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
            />
            <button
              type="button"
              className="lf-btn"
              aria-label="new random example"
              title="New example"
              onClick={() => setExampleSeed(Math.floor(Math.random() * 10000))}
            >
              New example
            </button>
          </div>
        </div>
      </div>

      <div className="lf-try__stage" aria-live="polite">
        {result.error && loading === false && (
          <div className="lf-error" role="alert">
            <strong>The server did not run this example.</strong>
            <p>{result.error}</p>
            <p className="lf-muted">If the server is down, start it with examiner_app/start.ps1.</p>
          </div>
        )}
        {data && !result.error && <PlaybackControls pb={pb} />}
        <Panels
          label="Try views"
          items={[
            {
              label: 'Answers',
              content: data && !result.error ? (
                <div className={loading ? 'lf-try__body is-loading' : 'lf-try__body'}>
                  <AnswersView data={data} arm={arm} />
                </div>
              ) : (
                skeleton
              ),
            },
            {
              label: 'Memory and gates',
              content: data && !result.error ? (
                <div className={loading ? 'lf-try__body is-loading' : 'lf-try__body'}>
                  <StatePlayer
                    response={data}
                    label={`${ARM_LABEL[arm]} · ${variant} · seed ${seed}`}
                    color={armColor(arm)}
                    playback={pb}
                    view="memory"
                    compact
                    showTokens={false}
                  />
                </div>
              ) : (
                skeleton
              ),
            },
            {
              label: 'Memory size',
              content: data && !result.error ? (
                <div className={loading ? 'lf-try__body is-loading' : 'lf-try__body'}>
                  <StatePlayer
                    response={data}
                    label={`${ARM_LABEL[arm]} · ${variant} · seed ${seed}`}
                    color={armColor(arm)}
                    playback={pb}
                    view="size"
                    compact
                  />
                </div>
              ) : (
                skeleton
              ),
            },
          ]}
        />
      </div>
    </div>
  );
}

function AnswersView({ data, arm }: { data: TryResponse; arm: Arm }) {
  const { example, model } = data;
  const blockFirst = isBlockFirst(example.pattern);
  const color = armColor(arm);
  const writes = example.pattern.split('').filter((c) => c === 'T').length;
  return (
    <div className="lf-answers">
      <p className="lf-muted lf-small">
        {example.tokens.length} positions · {writes} writes of the queried key · {blockFirst ? 'block-first' : 'interleaved'} ·{' '}
        {model.parameters.toLocaleString()} parameters
      </p>
      <TokenStrip tokens={example.tokens} roles={example.roles} compactFiller />
      <div className="lf-queries">
        {example.queries.map((q, i) => {
          const bars: BarDatum[] = q.top5.map((c) => ({
            label: `token ${c.token}`,
            value: c.prob,
            color: c.token === q.gold ? 'var(--c-good)' : c.token === q.prediction ? color : 'var(--border-strong)',
            note: [c.token === q.gold ? 'gold' : '', c.token === q.prediction ? 'predicted' : ''].filter(Boolean).join(', ') || undefined,
          }));
          const correct = q.prediction === q.gold;
          return (
            <article key={`${q.position}-${i}`} className="lf-query">
              <header>
                <strong>{q.kind === 'latest' ? 'Latest value' : 'Unrelated key'}</strong>
                <span className={correct ? 'lf-verdict is-ok' : 'lf-verdict is-bad'}>{correct ? 'correct' : 'wrong'}</span>
              </header>
              <p>
                The model {whichWriteText(q.which_write)}.
              </p>
              <AnimatedBars data={bars} max={1} format={(v) => `${(v * 100).toFixed(1)}%`} />
            </article>
          );
        })}
      </div>
      <WhatThisShows>the five most likely answers at each query; teal marks the correct one.</WhatThisShows>
    </div>
  );
}

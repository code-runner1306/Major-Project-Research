import { useState } from 'react';
import { TokenStrip } from '../../components/TokenStrip';
import { Term } from '../../components/Term';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Panels } from '../../components/Panels';
import { Remote, ModelSelect, Segmented, pct } from './common';
import { useEvaluate } from './data';
import {
  DEFAULT_MODEL,
  LAYOUTS,
  RULE_BY_ID,
  layoutsFor,
  pickDefault,
  whichWriteText,
  type EvaluateResponse,
  type ModelRef,
  type ShortcutLayout,
} from './types';

const REWRITES = [1, 2, 3, 4] as const;
const SHOWN = 3;
const SAMPLE = 16;

/** Section 6: choose a local model and an example, and see what the model answered and which rules agree with it. */
export function TryShortcut({ models }: { models: ModelRef[] }) {
  const [model, setModel] = useState<ModelRef>(() => pickDefault(models) ?? DEFAULT_MODEL);
  const [layoutChoice, setLayoutChoice] = useState<ShortcutLayout>('block_first');
  const [rewrites, setRewrites] = useState<number>(3);
  const [distance, setDistance] = useState(8);
  const [exampleSeed, setExampleSeed] = useState(7);
  const [index, setIndex] = useState(0);

  // A layout that does not exist for the chosen rewrite count falls back to block-first.
  const layout = layoutsFor(rewrites).includes(layoutChoice) ? layoutChoice : 'block_first';
  const ev = useEvaluate({
    ...model,
    layout,
    rewrites,
    distance,
    examples: SAMPLE,
    example_seed: exampleSeed,
    patch: 'none',
  });

  return (
    <div className="sc-try">
      <div className="sc-controls">
        <ModelSelect models={models} value={model} onChange={setModel} />
        <label className="sc-field">
          <span className="sc-field__label">Layout</span>
          <select value={layout} onChange={(e) => setLayoutChoice(e.target.value as ShortcutLayout)}>
            {LAYOUTS.filter((l) => layoutsFor(rewrites).includes(l.id)).map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </label>
        <div className="sc-field">
          <span className="sc-field__label">Rewrites of the key</span>
          <Segmented options={REWRITES.map((n) => ({ id: n, label: String(n) }))} value={rewrites} onChange={setRewrites} label="rewrites" />
        </div>
        <label className="sc-field">
          <span className="sc-field__label">Query distance: {distance} tokens</span>
          <input type="range" min={8} max={24} value={distance} onChange={(e) => setDistance(Number(e.target.value))} />
        </label>
        <div className="sc-field">
          <span className="sc-field__label">
            <Term term="seed" /> for the examples
          </span>
          <div className="sc-inline">
            <input
              type="number"
              min={0}
              value={exampleSeed}
              onChange={(e) => setExampleSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
            />
            <button type="button" className="sc-btn" onClick={() => setExampleSeed(Math.floor(Math.random() * 10000))}>
              Roll new examples
            </button>
          </div>
        </div>
      </div>

      <Remote state={ev} what="this example">
        {(r) => <TryBody r={r} index={index} onIndex={setIndex} />}
      </Remote>
    </div>
  );
}

function TryBody({ r, index, onIndex }: { r: EvaluateResponse; index: number; onIndex: (i: number) => void }) {
  const shown = r.examples.slice(0, SHOWN);
  if (shown.length === 0) return <p className="sc-muted">The server returned no examples for these settings.</p>;
  const at = Math.min(index, shown.length - 1);
  const ex = shown[at];
  const correct = ex.gold !== undefined ? ex.prediction === ex.gold : ex.which_write === 'latest';
  const matched = new Set(ex.rules_matched);

  return (
    <div className="sc-try__body">
      <div className="sc-try__top">
        <p className="sc-try__summary">
          Over {SAMPLE} fresh examples the model returned the latest value in <strong>{pct(r.accuracy_latest, 0)}</strong>. Example{' '}
          {at + 1} of {shown.length}.
        </p>
        <Segmented
          options={shown.map((_, i) => ({ id: i, label: `Example ${i + 1}` }))}
          value={at}
          onChange={onIndex}
          label="example"
        />
      </div>

      <div className={correct ? 'sc-verdict is-ok' : 'sc-verdict is-bad'}>
        <strong>The model answered token {ex.prediction}.</strong>{' '}
        {ex.gold !== undefined && <span>The answer key says token {ex.gold}. </span>}
        It {whichWriteText(ex.which_write)}.
      </div>

      <Panels
        label="Example views"
        items={[
          {
            label: 'Tokens',
            content: (
              <div className="sc-try__panel">
                <TokenStrip tokens={ex.tokens} roles={ex.roles} compactFiller />
                <WhatThisShows>
                  each square is one token, coloured by its role. Hover a square for its position. Runs of filler are collapsed.
                </WhatThisShows>
              </div>
            ),
          },
          {
            label: 'Which rules agree',
            content: (
              <div className="sc-try__panel">
                <ul className="sc-rule-list">
                  {Object.keys(RULE_BY_ID).map((id) => (
                    <li key={id} className={matched.has(id) ? 'is-match' : undefined}>
                      <span className="sc-tag">{matched.has(id) ? 'agrees' : 'differs'}</span>
                      <strong>{RULE_BY_ID[id].name}</strong>
                    </li>
                  ))}
                </ul>
                <WhatThisShows>
                  each rule predicts an answer from the same example. &ldquo;Agrees&rdquo; means the model&apos;s answer matches that rule&apos;s prediction.
                </WhatThisShows>
              </div>
            ),
          },
        ]}
      />
    </div>
  );
}

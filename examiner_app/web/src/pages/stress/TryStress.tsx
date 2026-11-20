import { useState } from 'react';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, Segmented, ModelSelect, NumberField, PairBars } from './common';
import { useInterventions, useStressModels, useTry } from './data';
import type { StressModel, TryIntervention, TryRequest, TryResponse } from './types';

const INTERVENTION_SHORT: Record<TryIntervention, string> = {
  none: 'As trained',
  beta_one_latest: 'Write strength 1 on newest',
  beta_zero_rewrite1: 'Block first replacement',
  key_align_latest: 'One address for the key',
  alpha_shared_mean: 'One forgetting rate',
};

function TryBody({ data }: { data: TryResponse }) {
  const rows = [
    { label: 'Newest value recalled', before: data.before.latest_accuracy, after: data.after.latest_accuracy },
    { label: 'Unrelated key kept', before: data.before.unrelated_retention, after: data.after.unrelated_retention },
    { label: 'First replacement returned', before: data.before.first_replacement_rate, after: data.after.first_replacement_rate },
    { label: 'Any obsolete value returned', before: data.before.stale_rate, after: data.after.stale_rate },
  ];
  const n = data.setup.examples;
  return (
    <div className="stress-try__result">
      <p className="stress-try__note">
        {data.setup.no_effect_expected ? (
          'Scalar models have one forgetting rate per head, so averaging over channels changes nothing. Choose a channel-wise model to see this change.'
        ) : (
          <>
            {data.changed_answers} of {n} answers changed: {data.fixed} fixed, {data.broken} broken.
          </>
        )}
        <span className="stress-try__count">
          {' · '}
          <span className="is-before">as trained</span> / <span className="is-after">with change</span>, {n} fresh examples
        </span>
      </p>
      <PairBars rows={rows} />
    </div>
  );
}

/** Controls and live before-and-after measures on fresh overwrite examples. */
export function TryVisual() {
  const models = useStressModels();
  const interventions = useInterventions();
  const [model, setModel] = useState<StressModel | null>(null);
  const [intervention, setIntervention] = useState<TryIntervention>('alpha_shared_mean');
  const [rewrites, setRewrites] = useState(3);
  const [distance, setDistance] = useState(8);
  const [examples, setExamples] = useState(32);
  const [exampleSeed, setExampleSeed] = useState(0);

  const list = models.data ?? [];
  const chosen = model ?? list.find((m) => m.init === 'default' && m.variant === 'channel-wise' && m.seed === 44) ?? list[0] ?? null;
  const body: TryRequest | null =
    chosen === null
      ? null
      : {
          init: chosen.init as TryRequest['init'],
          variant: chosen.variant,
          seed: chosen.seed as TryRequest['seed'],
          intervention,
          rewrites,
          distance,
          examples,
          example_seed: exampleSeed,
          show: 0,
        };
  const run = useTry(body, 260);
  const options = (interventions.data ?? []).map((i) => ({ id: i.id, label: INTERVENTION_SHORT[i.id] ?? i.label }));
  const detail = (interventions.data ?? []).find((i) => i.id === intervention)?.detail;

  return (
    <div className="stress-try">
      <div className="stress-try__row">
        {models.data && chosen && (
          <div className="stress-try__model">
            <ModelSelect models={list} value={chosen} onChange={(m) => setModel(m as StressModel)} />
          </div>
        )}
        <NumberField label="Example set" value={exampleSeed} min={0} max={99999} onChange={setExampleSeed} />
        <button type="button" className="stress-btn" onClick={() => setExampleSeed((s) => (s + 1) % 100000)}>
          New examples
        </button>
      </div>
      <Segmented label="Change to the model" value={intervention} onChange={setIntervention} options={options} />
      {detail && <p className="stress-try__detail">{detail}</p>}
      <div className="stress-try__row">
        <span className="stress-try__label">Rewrites</span>
        <Segmented
          label="Rewrites"
          value={rewrites}
          onChange={setRewrites}
          options={[1, 2, 3, 4].map((n) => ({ id: n, label: String(n) }))}
        />
        <span className="stress-try__label">Examples</span>
        <Segmented
          label="Examples"
          value={examples}
          onChange={setExamples}
          options={[16, 32, 64].map((n) => ({ id: n, label: String(n) }))}
        />
        <label className="stress-try__slider">
          <span className="stress-try__label">Tokens to query: {distance}</span>
          <input type="range" min={8} max={24} value={distance} onChange={(e) => setDistance(Number(e.target.value))} />
        </label>
      </div>
      <Remote state={run} what="the fresh examples" tall>
        {(data) => <TryBody data={data} />}
      </Remote>
      <WhatThisShows>fresh examples are run through the saved checkpoint, once as trained and once with the change.</WhatThisShows>
    </div>
  );
}

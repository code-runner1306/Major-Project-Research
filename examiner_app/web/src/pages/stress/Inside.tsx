import { useState } from 'react';
import { AnimatedLines } from '../../components/AnimatedLines';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, ModelSelect } from './common';
import { useStressModels, useTry } from './data';
import type { StressModel, TryIntervention, TryRequest, TryResponse, WhichWrite } from './types';

const INSIDE_OPTIONS: { id: TryIntervention; label: string }[] = [
  { id: 'none', label: 'As trained' },
  { id: 'beta_one_latest', label: 'Write strength 1 on newest' },
  { id: 'beta_zero_rewrite1', label: 'Block first replacement' },
  { id: 'key_align_latest', label: 'One address for the key' },
  { id: 'alpha_shared_mean', label: 'One forgetting rate' },
];

const WHICH_TEXT: Record<WhichWrite, string> = {
  latest: 'the newest value',
  initial: 'the first value written',
  first_rewrite: 'the first replacement',
  second_rewrite: 'the second replacement',
  third_rewrite: 'the third replacement',
  other: 'a value never written',
};

/** The memory cards in write order, tagged with the write each run's answer came from. */
function MemoryCards({ data }: { data: TryResponse }) {
  const { writes, queries } = data.example;
  const latest = queries.find((q) => q.kind === 'latest');
  const targets = writes.filter((w) => w.is_target);
  const newestTarget = targets.length ? targets[targets.length - 1].slot : -1;
  return (
    <div className="stress-cards" aria-label="memory in write order">
      {writes.map((w) => (
        <div key={w.slot} className={w.is_target ? 'stress-card is-target' : 'stress-card'}>
          <div className="stress-card__slot">write {w.slot}</div>
          <div className="stress-card__key">key {w.key}</div>
          <div className="stress-card__value">value {w.value}</div>
          <div className="stress-card__tags">
            {w.is_target && <span className="stress-tag">same key</span>}
            {w.slot === newestTarget && <span className="stress-tag is-newest">newest</span>}
            {latest?.before.picked_slot === w.slot && <span className="stress-tag is-before">as trained</span>}
            {latest?.after.picked_slot === w.slot && <span className="stress-tag is-after">with change</span>}
          </div>
        </div>
      ))}
    </div>
  );
}

function AnswerLine({ data }: { data: TryResponse }) {
  const latest = data.example.queries.find((q) => q.kind === 'latest');
  if (!latest) return null;
  return (
    <p className="stress-answer">
      The newest value for this key is <strong>{latest.gold}</strong>. As trained answers <strong>{latest.before.prediction}</strong>{' '}
      ({WHICH_TEXT[latest.before.which ?? 'other']}). With the change it answers <strong>{latest.after.prediction}</strong> (
      {WHICH_TEXT[latest.after.which ?? 'other']}).
    </p>
  );
}

type Quantity = 'alpha' | 'beta' | 'memory';

/** One gate or memory quantity over the example, as trained against with the change. */
function GateChart({ data, layer, quantity }: { data: TryResponse; layer: number; quantity: Quantity }) {
  const b = data.gates.before.find((l) => l.layer === layer);
  const a = data.gates.after.find((l) => l.layer === layer);
  if (!b || !a) return null;
  const pick = (l: typeof b) => (quantity === 'alpha' ? l.alpha_mean : quantity === 'beta' ? l.beta : l.state_norm);
  return (
    <AnimatedLines
      series={[
        { name: 'As trained', ys: pick(b), color: 'var(--text-muted)' },
        { name: 'With change', ys: pick(a), color: 'var(--accent)', dashed: true },
      ]}
      yLabel={quantity === 'alpha' ? 'α, mean over channels' : quantity === 'beta' ? 'β' : 'size of memory'}
      xLabel="token position"
      yMax={quantity === 'memory' ? undefined : 1}
    />
  );
}

function InsideBody({ data, layer }: { data: TryResponse; layer: number }) {
  return (
    <Panels
      label="Inside the model"
      items={[
        {
          label: 'Memory',
          content: (
            <>
              <MemoryCards data={data} />
              <AnswerLine data={data} />
            </>
          ),
        },
        { label: 'Decay gate (α)', content: <GateChart data={data} layer={layer} quantity="alpha" /> },
        { label: 'Write gate (β)', content: <GateChart data={data} layer={layer} quantity="beta" /> },
        { label: 'Memory size', content: <GateChart data={data} layer={layer} quantity="memory" /> },
      ]}
    />
  );
}

export function InsideVisual() {
  const models = useStressModels();
  const [model, setModel] = useState<StressModel | null>(null);
  const [intervention, setIntervention] = useState<TryIntervention>('alpha_shared_mean');
  const [layer, setLayer] = useState(1);

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
          rewrites: 3,
          distance: 8,
          examples: 32,
          example_seed: 0,
          show: 0,
        };
  const run = useTry(body);

  return (
    <div className="stress-inside">
      <div className="stress-inside__controls">
        {models.data && chosen && (
          <div className="stress-inside__model">
            <ModelSelect models={list} value={chosen} onChange={(m) => setModel(m as StressModel)} compact />
          </div>
        )}
        <label className="stress-field">
          <select aria-label="Change to the model" value={intervention} onChange={(e) => setIntervention(e.target.value as TryIntervention)}>
            {INSIDE_OPTIONS.map((o) => (
              <option key={o.id} value={o.id}>
                {o.label}
              </option>
            ))}
          </select>
        </label>
        <label className="stress-field">
          <select aria-label="Layer" value={layer} onChange={(e) => setLayer(Number(e.target.value))}>
            {[1, 2, 3, 4].map((n) => (
              <option key={n} value={n}>
                {`Layer ${n}`}
              </option>
            ))}
          </select>
        </label>
      </div>
      <Remote state={run} what="the live example" tall>
        {(data) => <InsideBody data={data} layer={layer} />}
      </Remote>
      <WhatThisShows>one fixed example, same tokens in both runs; grey is as trained, red dashed is with the change.</WhatThisShows>
    </div>
  );
}

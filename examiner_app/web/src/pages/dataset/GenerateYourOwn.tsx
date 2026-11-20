import { useState } from 'react';
import { Term } from '../../components/Term';
import { WhatThisShows } from '../../components/WhatThisShows';
import {
  ExampleError,
  OriginBadge,
  ScoredStrip,
  SeedControl,
  Skeleton,
  WaitingForServer,
  defaultParams,
  useExample,
  useGenerators,
} from './shared';
import type { DatasetExample, GeneratorInfo, ParamSpec, ParamValue } from './types';

// Section 6: pick a generator, set its parameters from the server's schema, and read the example back.

function humanise(name: string): string {
  return name.replace(/_/g, ' ');
}

function rangeText(spec: ParamSpec): string {
  if (spec.type === 'choice') return (spec.choices ?? []).join(', ');
  if (spec.min !== undefined && spec.max !== undefined) return `${spec.min} to ${spec.max}`;
  if (spec.min !== undefined) return `at least ${spec.min}`;
  if (spec.max !== undefined) return `at most ${spec.max}`;
  return 'any whole number';
}

function ParamField({
  name,
  spec,
  value,
  onChange,
}: {
  name: string;
  spec: ParamSpec;
  value: ParamValue;
  onChange: (value: ParamValue) => void;
}) {
  const id = `ds-param-${name}`;
  const hint = `Allowed: ${rangeText(spec)}. Default ${String(spec.default)}.`;
  if (spec.type === 'choice') {
    return (
      <label className="ds-field" htmlFor={id}>
        <span className="ds-label">{humanise(name)}</span>
        <select id={id} value={String(value)} onChange={(e) => onChange(e.target.value)} aria-describedby={`${id}-hint`}>
          {(spec.choices ?? []).map((c) => (
            <option key={c} value={c}>
              {c}
            </option>
          ))}
        </select>
        <span className="ds-field__hint" id={`${id}-hint`}>{hint}</span>
      </label>
    );
  }
  return (
    <label className="ds-field" htmlFor={id}>
      <span className="ds-label">{humanise(name)}</span>
      <input
        id={id}
        type="number"
        step={1}
        min={spec.min}
        max={spec.max}
        value={String(value)}
        onChange={(e) => onChange(Number(e.target.value))}
        aria-describedby={`${id}-hint`}
      />
      <span className="ds-field__hint" id={`${id}-hint`}>{hint}</span>
    </label>
  );
}

/** Reading of the example: the notes sentences, scored answers, reserved-token legend, and raw JSON. */
function Reading({ ex, pending }: { ex: DatasetExample; pending: boolean }) {
  const [raw, setRaw] = useState(false);
  const reserved = Object.entries(ex.reserved_tokens);
  const scored = ex.scored;

  return (
    <div className="ds-try__reading">
      <ScoredStrip ex={ex} pending={pending} />
      <WhatThisShows>
        each square is one token, coloured by its role. Outlined squares are the scored answers. The legend under the strip names each role.
      </WhatThisShows>

      <h4 className="ds-h4">What this example says</h4>
      {ex.notes.length > 0 ? (
        <ul className="ds-list">
          {ex.notes.map((note, i) => (
            <li key={i}>{note}</li>
          ))}
        </ul>
      ) : (
        <p className="ds-muted">The server gave no notes for this example.</p>
      )}

      <h4 className="ds-h4">Scored answers</h4>
      <p>
        {scored.length === 0
          ? 'This example has no scored answers.'
          : `${scored.length} scored position${scored.length === 1 ? '' : 's'}. The model is graded only at these positions.`}
      </p>
      {scored.length > 0 && (
        <ul className="ds-list ds-list--inline">
          {scored.slice(0, 8).map((s) => (
            <li key={s.position}>
              position {s.position}: answer {s.answer}
              {s.kind ? ` (${s.kind})` : ''}
            </li>
          ))}
          {scored.length > 8 && <li>and {scored.length - 8} more</li>}
        </ul>
      )}

      <h4 className="ds-h4">Reserved tokens in this generator</h4>
      {reserved.length > 0 ? (
        <ul className="ds-list">
          {reserved.map(([token, meaning]) => (
            <li key={token}>
              Token {token} means: {meaning}.
            </li>
          ))}
        </ul>
      ) : (
        <p className="ds-muted">This generator reserves no token numbers.</p>
      )}
      <p className="ds-muted">Different generators reserve different numbers, so 0 can mean one thing here and something else elsewhere.</p>

      <button type="button" className="ds-btn" aria-expanded={raw} onClick={() => setRaw((r) => !r)}>
        {raw ? 'Hide raw JSON' : 'Show raw JSON'}
      </button>
      {raw && <pre className="ds-json">{JSON.stringify(ex, null, 2)}</pre>}
    </div>
  );
}

/** Parameter form and result for one generator. Keyed by generator id, so switching generator resets it. */
function GeneratorRun({ info }: { info: GeneratorInfo }) {
  const [params, setParams] = useState<Record<string, ParamValue>>(() => defaultParams(info));
  const [seed, setSeed] = useState(0);
  const ex = useExample({ generator: info.id, seed, params });

  return (
    <>
      <div className="ds-try__controls">
        <SeedControl seed={seed} onChange={setSeed} />
        {Object.entries(info.params).map(([name, spec]) => (
          <ParamField
            key={name}
            name={name}
            spec={spec}
            value={params[name] ?? spec.default}
            onChange={(value) => setParams((p) => ({ ...p, [name]: value }))}
          />
        ))}
      </div>
      <div className="ds-try__result" aria-live="polite">
        {ex.error && <ExampleError error={ex.error} what="This example could not be built." />}
        {ex.waiting && <WaitingForServer />}
        {!ex.data && !ex.error && !ex.waiting && <Skeleton label="generating an example" tall />}
        {ex.data && <Reading ex={ex.data} pending={ex.pending} />}
      </div>
    </>
  );
}

export function GenerateYourOwn() {
  const gens = useGenerators();
  const [genId, setGenId] = useState('overwrite');

  if (gens.pending) return <Skeleton label="loading the generator list" tall />;
  if (gens.waiting) return <WaitingForServer />;
  if (gens.error) return <ExampleError error={gens.error} what="The generator list could not be loaded." />;

  const list = gens.data ?? [];
  const info = list.find((g) => g.id === genId) ?? list[0];
  if (!info) return <p className="ds-muted">The server lists no generators.</p>;

  return (
    <div className="ds-try">
      <p className="ds-muted">
        Pick a generator and its settings. The example is built on the server with the same code the experiments use. Each generator uses its own
        reserved tokens; the reading under the strip says what they mean here. Give the same <Term term="seed" /> and you get the same example back.
      </p>
      <div className="ds-try__pick">
        <label className="ds-field">
          <span className="ds-label">Generator</span>
          <select value={info.id} onChange={(e) => setGenId(e.target.value)}>
            {list.map((g) => (
              <option key={g.id} value={g.id}>
                {g.name}
                {g.live ? '' : ' (not offline)'}
              </option>
            ))}
          </select>
        </label>
        <p className="ds-try__source">
          Source: <code>{info.source}</code> <OriginBadge origin={info.origin} />
        </p>
      </div>
      <GeneratorRun key={info.id} info={info} />
    </div>
  );
}

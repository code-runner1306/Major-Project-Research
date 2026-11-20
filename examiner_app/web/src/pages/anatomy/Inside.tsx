import { useState } from 'react';
import { ColumnBars } from './ColumnBars';
import { MoreDetail } from '../../components/MoreDetail';
import { Panels } from '../../components/Panels';
import { TokenStrip } from '../../components/TokenStrip';
import { Unavailable } from '../../components/Unavailable';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, Segmented } from '../shortcut/common';
import { RoleLegend } from '../dataset/shared';
import { useAnatomyExample } from './data';
import { pct } from './Results';
import type { AnatomyExample, CellId, Summary } from './types';

const KEY_ROLES = new Set(['key', 'value']);
const QUERY_ROLES = new Set(['query_key', 'mask']);

interface Block {
  name: string;
  start: number;
  end: number; // inclusive
  tokens: number[];
  roles: string[];
}

/** Splits the row into table, haystack and queries by role, so the brackets are read from the real example. */
export function blocksOf(ex: AnatomyExample): { table: Block; haystack: Block; queries: Block } {
  const n = ex.tokens.length;
  let t = 0;
  while (t < n && KEY_ROLES.has(ex.roles[t])) t++;
  let q = n;
  while (q > 0 && QUERY_ROLES.has(ex.roles[q - 1])) q--;
  const make = (name: string, start: number, end: number): Block => ({
    name,
    start,
    end,
    tokens: ex.tokens.slice(start, end),
    roles: ex.roles.slice(start, end),
  });
  return {
    table: make('table', 0, t),
    haystack: make('haystack', t, q),
    queries: make('queries', q, n),
  };
}

/** One run of the real generator drawn as three blocks, widths in proportion to their lengths. */
function ConstructionDiagram({ ex }: { ex: AnatomyExample }) {
  const { table, haystack, queries } = blocksOf(ex);
  const collision = ex.experiment === 'E5';
  const pairs = table.tokens.length / 2;
  const pos = (b: Block) => `positions ${b.start}–${b.end - 1}`;
  return (
    <div className="anat-build">
      <div className="anat-build__row">
        <div className="anat-build__block anat-build__block--table" >
          <div className="anat-build__name">
            1 · Table: {pairs} key–value pairs <span className="anat-muted">{pos(table)}</span>
          </div>
          <TokenStrip tokens={table.tokens} roles={table.roles} />
        </div>
        <div className="anat-build__block anat-build__block--haystack" >
          <div className="anat-build__name">
            2 · {collision ? 'Haystack: decoy pairs reusing the table keys' : 'Haystack: distractor tokens, no keys'}{' '}
            <span className="anat-muted">{pos(haystack)}</span>
          </div>
          <TokenStrip tokens={haystack.tokens} roles={haystack.roles} compactFiller />
        </div>
        <div className="anat-build__block anat-build__block--queries" >
          <div className="anat-build__name">
            3 · Queries: key, then MASK <span className="anat-muted">{pos(queries)}</span>
          </div>
          <TokenStrip tokens={queries.tokens} roles={queries.roles} />

        </div>
      </div>
      <p className="anat-build__answers">
        answers (hidden from the model): {ex.scored.map((x) => `slot ${x.position} = ${x.answer}`).join(' · ')}
      </p>
    </div>
  );
}

/** The real generator's own plain-English notes for this example. */
function GeneratorNotes({ ex }: { ex: AnatomyExample }) {
  return (
    <ul className="anat-notes">
      {ex.notes.map((n, i) => (
        <li key={i}>{n}</li>
      ))}
    </ul>
  );
}

function BuildVisual() {
  const [collision, setCollision] = useState<'E4' | 'E5'>('E4');
  const [seed, setSeed] = useState(0);
  const example = useAnatomyExample(collision, seed);
  return (
    <div className="anat-inside__build">
      <div className="anat-controls">
        <Segmented
          label="Haystack"
          value={collision}
          onChange={setCollision}
          options={[
            { id: 'E4', label: 'Distractor haystack' },
            { id: 'E5', label: 'Colliding keys (switch on)' },
          ]}
        />
        <div className="anat-seed">
          <label>
            Seed <input type="number" min={0} value={seed} onChange={(e) => setSeed(Math.max(0, Math.floor(Number(e.target.value) || 0)))} />
          </label>
          <button type="button" onClick={() => setSeed(Math.floor(Math.random() * 10000))}>
            Roll the dice
          </button>
        </div>
      </div>
      <Remote state={example} what="this example" tall>
        {(ex) => (
          <>
            <ConstructionDiagram ex={ex} />
            <MoreDetail>
              <GeneratorNotes ex={ex} />
              <RoleLegend roles={[...new Set(ex.roles)]} />
              <p className="anat-muted">
                Built by the upstream generator (make_long_retrieval_batch) through the examiner&apos;s dataset module, not re-typed here. The switch
                changes the haystack; for one seed the table is the same in both settings.
              </p>
            </MoreDetail>
          </>
        )}
      </Remote>
      <WhatThisShows>one generated example: table, then haystack, then the masked queries.</WhatThisShows>
    </div>
  );
}

/** Saved accuracy per query position is not in any result file: the probes pool all answer slots. */
function PerPositionVisual() {
  return (
    <div className="anat-inside__position">
      <Unavailable reason="The saved results record one accuracy pooled over the four answer slots, so accuracy per query position was not saved." />
      <WhatThisShows>nothing per position exists on disk to plot.</WhatThisShows>
    </div>
  );
}

const SHORT_INTERVENTION: Record<string, string> = {
  original: 'Unchanged',
  fixed_channel_mean: 'Channel average fixed',
  token_channel_average: 'Per-token average',
  fixed_scalar_mean: 'One constant',
};

/** Decay interventions (saved): the trained channel-wise models with their decay replaced, mean over seeds. */
function InterventionVisual({ summary }: { summary: Summary }) {
  const ids: CellId[] = ['E1', 'E4', 'E5'];
  const groups = ids.flatMap((id) => {
    const cell = summary.cells.find((c) => c.id === id);
    const iv = summary.interventions[id];
    if (!cell || !iv) return [];
    return [{
      title: <>{cell.title} <span className="anat-muted">({iv.n} seeds)</span></>,
      bars: Object.keys(summary.intervention_labels).map((key) => ({
        label: SHORT_INTERVENTION[key] ?? key,
        value: iv.means[key] ?? 0,
        color: key === 'original' ? 'var(--c-channel)' : 'var(--text-muted)',
      })),
    }];
  });
  return (
    <div className="anat-interventions">
      <ColumnBars groups={groups} format={(v) => pct(v)} />
      <WhatThisShows>trained channel-wise models with their decay replaced, averaged over seeds.</WhatThisShows>
    </div>
  );
}

export function InsideVisual({ summary }: { summary: Summary }) {
  return (
    <div className="anat-inside">
      <Panels
        label="Inside"
        items={[
          { label: 'How a test is built', content: <BuildVisual /> },
          { label: 'Decay replaced', content: <InterventionVisual summary={summary} /> },
          { label: 'Per query position', content: <PerPositionVisual /> },
        ]}
      />
    </div>
  );
}

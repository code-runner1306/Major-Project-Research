import { useState } from 'react';
import { Panels } from '../../components/Panels';
import { AnimatedBars } from '../../components/AnimatedBars';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, Segmented, SignedBars } from './common';
import {
  MQAR_IDS,
  shortSettingLabel,
  OVERWRITE_IDS,
  PATCH_LABEL,
  START_LABEL,
  cellLabel,
  cellOf,
  gateMeansFor,
  patchById,
  pct,
  settingsFor,
} from './derive';
import { useStressSummary } from './data';
import type { StartLabel, StressFamily, StressSummary } from './types';

const OVERWRITE_SHORT: Record<string, string> = {
  baseline: 'Standard',
  more_keys: 'More keys',
  farther_query: 'Query further',
  more_rewrites: 'Three rewrites',
};
const MQAR_SHORT: Record<string, string> = {
  baseline: 'Standard',
  more_keys: 'More keys',
  longer_context: 'Longer',
  longer_more_keys: 'Longer, more keys',
};
const GATE_TESTS = [
  { id: 'baseline', label: 'As trained' },
  { id: 'shared_mean', label: 'One forgetting rate (α forced)' },
  { id: 'static', label: 'Fixed per-channel rates' },
  { id: 'shuffle_mean', label: 'Channels shuffled' },
] as const;

function OverwriteView({ s }: { s: StressSummary }) {
  const [cell, setCell] = useState('more_rewrites');
  const rows = settingsFor(s, 'overwrite', cell);
  return (
    <section className="stress-panel">
      <h3 className="stress-h3">Newest-value recall: {cellLabel(cellOf(s, 'overwrite', cell))}</h3>
      <Segmented
        label="Overwrite condition"
        value={cell}
        onChange={setCell}
        options={OVERWRITE_IDS.map((id) => ({ id, label: OVERWRITE_SHORT[id] }))}
      />
      <AnimatedBars
        max={1}
        format={(v) => pct(v, 1)}
        data={rows.map((r) => ({
          label: shortSettingLabel(r.initialization, r.variant),
          value: r.latest ?? 0,
          note: `obsolete ${pct(r.stale ?? 0, 0)}`,
        }))}
      />
      <WhatThisShows>each bar is the mean over three seeds; “obsolete” means an earlier value was returned.</WhatThisShows>
    </section>
  );
}

function MqarView({ s }: { s: StressSummary }) {
  const [cell, setCell] = useState('longer_more_keys');
  const rows = settingsFor(s, 'mqar', cell);
  return (
    <section className="stress-panel">
      <h3 className="stress-h3">Exact recall: {cellLabel(cellOf(s, 'mqar', cell))}</h3>
      <Segmented
        label="Recall condition"
        value={cell}
        onChange={setCell}
        options={MQAR_IDS.map((id) => ({ id, label: MQAR_SHORT[id] }))}
      />
      <AnimatedBars
        max={1}
        format={(v) => pct(v, 1)}
        data={rows.map((r) => ({ label: shortSettingLabel(r.initialization, r.variant), value: r.accuracy }))}
      />
      <WhatThisShows>mean over three seeds; one default scalar seed alone drops to about half at eight keys.</WhatThisShows>
    </section>
  );
}

function GateView({ s }: { s: StressSummary }) {
  const [family, setFamily] = useState<StressFamily>('overwrite');
  const [init, setInit] = useState<StartLabel>('default');
  const means = gateMeansFor(s, family, init);
  return (
    <section className="stress-panel">
      <h3 className="stress-h3">Channel-wise models with the gates overridden, {START_LABEL[init].toLowerCase()}</h3>
      <div className="stress-panel__controls">
        <Segmented
          label="Task"
          value={family}
          onChange={setFamily}
          options={[
            { id: 'mqar', label: 'Multi-key recall' },
            { id: 'overwrite', label: 'Overwrite' },
          ]}
        />
        <Segmented
          label="Start"
          value={init}
          onChange={setInit}
          options={[
            { id: 'default', label: 'Default start' },
            { id: 'mean-decay', label: 'Mean-decay start' },
          ]}
        />
      </div>
      <AnimatedBars
        max={1}
        format={(v) => pct(v, 1)}
        data={GATE_TESTS.map((t) => ({
          label: t.label,
          value: means ? (means[t.id] as number) : 0,
          color: t.id === 'baseline' ? 'var(--c-good)' : 'var(--accent)',
        }))}
      />
      <WhatThisShows>accuracy averaged over three seeds, with the weights unchanged; a bar below “As trained” is lost accuracy.</WhatThisShows>
    </section>
  );
}

function PatchView({ s }: { s: StressSummary }) {
  const ids = ['beta_one_latest', 'beta_zero_rewrite1', 'key_align_latest', 'control_beta_one_unrelated'];
  const rows = ids
    .map((id) => patchById(s, id))
    .filter((p): p is NonNullable<typeof p> => p !== undefined)
    .map((p) => ({
      label: PATCH_LABEL[p.id] ?? p.id,
      value: p.latest_change,
      note: `better in ${p.improved}, worse in ${p.worse} of ${p.models}`,
    }));
  const range = Math.max(0.05, ...rows.map((r) => Math.abs(r.value))) * 1.1;
  return (
    <section className="stress-panel">
      <h3 className="stress-h3">Newest-value recall with three rewrites, change from as trained</h3>
      <SignedBars rows={rows} range={range} />
      <WhatThisShows>
        {`change in points, mean over ${s.write_patches.baseline.models} models; a change of ${s.write_patches.threshold_points} points or more counts as better or worse.`}
      </WhatThisShows>
    </section>
  );
}

function ResultsBody({ s }: { s: StressSummary }) {
  return (
    <div className="stress-results">
      <Panels
        label="Stress results"
        items={[
          { label: 'Overwrite', content: <OverwriteView s={s} /> },
          { label: 'Multi-key recall', content: <MqarView s={s} /> },
          { label: 'Gates overridden', content: <GateView s={s} /> },
          { label: 'Write-gate patches', content: <PatchView s={s} /> },
        ]}
      />
    </div>
  );
}

export function ResultsVisual() {
  const summary = useStressSummary();
  return (
    <Remote state={summary} what="the saved stress results" tall>
      {(s) => <ResultsBody s={s} />}
    </Remote>
  );
}

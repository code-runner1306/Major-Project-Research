import { useState } from 'react';
import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { pct, Segmented } from './common';
import type { InsideFile, InsideShortcut, InsideZoology, SlotCount } from './data';

const SEEDS = [
  { id: 44, label: 'Seed 44 model' },
  { id: 43, label: 'Seed 43 model' },
] as const;

/** Section 5: where the answers come from. Both views are slot histograms from real checkpoints (controls_export.py). */
export function InsideVisual({ inside }: { inside: InsideFile }) {
  const [seed, setSeed] = useState<44 | 43>(44);
  const zoology = inside.zoology.find((z) => z.init === 'default' && z.variant === 'scalar' && z.seed === seed && z.layout === 'shift_1');
  const zoologyStandard = inside.zoology.find((z) => z.init === 'default' && z.variant === 'scalar' && z.seed === seed && z.layout === 'shift_0');
  const shortcut = inside.shortcut.find((s) => s.init === 'default' && s.variant === 'scalar' && s.seed === seed);

  return (
    <div className="ctl-inside">
      <div className="ctl-inside__top">
        <Segmented options={[...SEEDS]} value={seed} onChange={setSeed} label="which model" />
      </div>
      <Panels
        label="Where answers come from"
        items={[
          {
            label: 'Zoology model, one blank token',
            content: zoology ? (
              <ZoologyView entry={zoology} standard={zoologyStandard} />
            ) : (
              <p className="ctl-muted">No saved histogram for this model.</p>
            ),
          },
          {
            label: 'Shortcut model, block-first',
            content: shortcut ? <ShortcutView entry={shortcut} /> : <p className="ctl-muted">No saved histogram for this model.</p>,
          },
        ]}
      />
    </div>
  );
}

function ZoologyView({ entry, standard }: { entry: InsideZoology; standard?: InsideZoology }) {
  const own: BarDatum[] = [];
  const other: BarDatum[] = [];
  for (const row of entry.slot_histogram as SlotCount[]) {
    if (row.slot === null) {
      other.push({ label: 'no memory slot', value: row.count });
      continue;
    }
    own.push({ label: `slot ${row.slot}`, value: row.from_queried_key ?? 0 });
    other.push({ label: `slot ${row.slot}`, value: row.from_other_key ?? 0, color: 'var(--accent)' });
  }
  const otherShare = entry.n_answers ? entry.errors.other_pair / entry.n_answers : 0;
  const anchoredShare = entry.n_answers ? entry.errors.position_anchored / entry.n_answers : 0;
  return (
    <div className="ctl-inside__view">
      <p className="ctl-result__caption">
        Answers by memory slot, {entry.n_answers} answers. Accuracy {pct(entry.accuracy, 1)} with one blank token
        {standard ? `, ${pct(standard.accuracy, 1)} on the standard layout` : ''}.
      </p>
      <div className="ctl-two-col">
        <div>
          <p className="ctl-result__caption">Its own key&apos;s value</p>
          <AnimatedBars data={own} format={(v) => String(Math.round(v))} />
        </div>
        <div>
          <p className="ctl-result__caption">Another key&apos;s value</p>
          <AnimatedBars data={other} format={(v) => String(Math.round(v))} />
        </div>
      </div>
      <WhatThisShows>
        {pct(otherShare, 1)} of answers returned another key&apos;s value; {pct(anchoredShare, 1)} returned the position-anchored token.
      </WhatThisShows>
    </div>
  );
}

function ShortcutView({ entry }: { entry: InsideShortcut }) {
  const bars: BarDatum[] = entry.pick_histogram.map((row) => {
    if (row.slot === null) return { label: 'no memory slot', value: row.count };
    const label = row.is_latest_target
      ? `slot ${row.slot}: newest value`
      : row.is_target
        ? `slot ${row.slot}: earlier write, same key`
        : `slot ${row.slot}: another key`;
    return { label, value: row.count, color: row.is_latest_target ? undefined : 'var(--accent)' };
  });
  return (
    <div className="ctl-inside__view">
      <p className="ctl-result__caption">
        Answers by memory slot, {entry.n_answers} answers. Newest value returned {pct(entry.accuracy_latest, 1)} of the time.
      </p>
      <AnimatedBars data={bars} format={(v) => String(Math.round(v))} />
      <WhatThisShows>
        {entry.first_rewrite_rate === null
          ? 'No wrong answers on this run.'
          : `${pct(entry.first_rewrite_rate, 1)} of its wrong answers were the first replacement.`}
      </WhatThisShows>
    </div>
  );
}

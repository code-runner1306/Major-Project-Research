import { AnimatedBars } from '../../components/AnimatedBars';
import { AnimatedLines } from '../../components/AnimatedLines';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { pct } from './common';
import { designLabel, modelLabel, type ControlsSummary, type ZoologyDesign } from './data';

const SHIFT_LAYOUTS = ['shift_0', 'shift_1', 'shift_2', 'shift_4', 'shift_8', 'shift_16'] as const;
const SHIFT_X = [0, 1, 2, 4, 8, 16];
const KATA_LENGTHS = ['64', '128', '256', '384'];
const KATA_X = [64, 128, 256, 384];
const REWRITE_CELLS: { cell: string; label: string }[] = [
  { cell: 'one_pass_K32', label: 'one write pass' },
  { cell: 'two_pass_K32', label: 'two passes (trained)' },
  { cell: 'three_pass_K32', label: 'three passes' },
  { cell: 'four_pass_K32', label: 'four passes' },
  { cell: 'partial_third_K32', label: 'third pass on half the keys' },
];

/** Section 4: four saved-result views. Every number comes from the summary, which reads the result files. */
export function ResultsVisual({ summary }: { summary: ControlsSummary }) {
  const z = summary.zoology;
  const k = summary.kata;
  const mainDesigns = z.designs.filter((d) => d.init !== 'published');

  const designLines = mainDesigns.map((d: ZoologyDesign) => ({
    name: designLabel(d.init, d.variant),
    ys: SHIFT_LAYOUTS.map((l) => 100 - d.accuracy[l] * 100),
  }));

  const perModel = z.models
    .filter((m) => m.group === 'main' || m.group === 'published')
    .map((m) => ({
      label: modelLabel(m),
      value: m.accuracy.shift_1,
      color: m.shortcut ? 'var(--accent)' : undefined,
      note: m.shortcut ? 'drops on the shifted layout' : undefined,
    }));

  const sel = k.selected_learning_rate;
  const rewriteBars = REWRITE_CELLS.map((r) => ({
    label: r.label,
    value: k.cells[r.cell]?.[sel] ?? 0,
  }));
  const designGroups = [...new Map(z.models.map((m) => [designLabel(m.init, m.variant), m.init + m.variant] as const)).keys()].map((title) => {
    const members = z.models.filter((m) => designLabel(m.init, m.variant) === title);
    return {
      title,
      data: members.map((m) => ({
        label: `seed ${m.seed}`,
        value: m.accuracy.shift_1,
        color: m.shortcut ? 'var(--accent)' : undefined,
      })),
    };
  });
  const lowestRewrite = Math.min(...rewriteBars.map((b) => b.value));
  const others = perModel.filter((m) => m.color === undefined);
  const lowestOther = others.length ? Math.min(...others.map((m) => m.value)) : null;

  return (
    <Panels
      label="Results views"
      items={[
        {
          label: 'Zoology, shifted layouts',
          content: (
            <div className="ctl-result">
              <AnimatedLines
                series={designLines}
                xs={SHIFT_X}
                yLabel="accuracy lost (points)"
                xLabel="blank tokens before the table"
              />
              <WhatThisShows>Points of accuracy lost, averaged over the three seeds of each decay design.</WhatThisShows>
            </div>
          ),
        },
        {
          label: 'Zoology, each model',
          content: (
            <div className="ctl-result">
              <p className="ctl-result__caption">Accuracy with one blank token before the table. Red bars drop on this layout.</p>
              <div className="ctl-design-bars">
                {designGroups.map((g) => (
                  <div key={g.title} className="ctl-design-bars__group">
                    <span className="ctl-design-bars__title">{g.title}</span>
                    <AnimatedBars data={g.data} max={1} format={(v) => pct(v, 1)} />
                  </div>
                ))}
              </div>
              <WhatThisShows>{lowestOther === null ? 'Red bars drop on this layout.' : `Every other model stays at or above ${pct(lowestOther, 1)} on this layout.`}</WhatThisShows>
            </div>
          ),
        },
        {
          label: 'KATA, replication',
          content: (
            <div className="ctl-result">
              <AnimatedLines
                series={[
                  { name: 'this run', ys: KATA_LENGTHS.map((x) => (k.replication.ours[x] ?? 0) * 100) },
                  { name: 'paper', ys: KATA_LENGTHS.map((x) => (k.replication.paper[x] ?? 0) * 100), dashed: true },
                ]}
                xs={KATA_X}
                yMax={100}
                yLabel="accuracy (%)"
                xLabel="keys in the sequence"
              />
              <WhatThisShows>{k.readings.replication}</WhatThisShows>
            </div>
          ),
        },
        {
          label: 'KATA, rewrite test',
          content: (
            <div className="ctl-result">
              <p className="ctl-result__caption">Accuracy with 32 keys, selected learning rate {sel.replace('lr_', '')}, by number of write passes.</p>
              <AnimatedBars data={rewriteBars} max={1} format={(v) => pct(v, 1)} />
              <WhatThisShows>{k.readings.shortcut} Lowest pattern here: {pct(lowestRewrite, 1)}.</WhatThisShows>
            </div>
          ),
        },
      ]}
    />
  );
}

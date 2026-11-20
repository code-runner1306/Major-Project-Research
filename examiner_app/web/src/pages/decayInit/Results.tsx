import { AnimatedBars } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, conditionLabel, pct, pointsSigned } from './common';
import { useDecaySummary } from './data';
import type { Block, DecayInit, MeasureRow, Summary } from './types';

const ORDER: { init: DecayInit; variant: 'scalar' | 'channel-wise' }[] = [
  { init: 'default', variant: 'scalar' },
  { init: 'default', variant: 'channel-wise' },
  { init: 'mean-decay', variant: 'scalar' },
  { init: 'mean-decay', variant: 'channel-wise' },
  { init: 'joint', variant: 'scalar' },
  { init: 'joint', variant: 'channel-wise' },
];

/** One saved measure: available conditions as bars, then each seed and paired granularity gaps. */
function MeasureView({
  rows,
  gaps,
  seeds,
  caption,
}: {
  rows: MeasureRow[];
  gaps: { default: Block; mean_decay: Block; joint?: Block; interaction?: Block } | undefined;
  seeds: number[];
  caption: string;
}) {
  const orderedRows = ORDER.flatMap((o) => {
    const row = rows.find((x) => x.init === o.init && x.variant === o.variant);
    return row ? [{ ...o, row }] : [];
  });
  const bars = orderedRows.map(({ init, variant, row }) => ({
    label: `${variant === 'scalar' ? 'Scalar' : 'Channel-wise'}, ${init === 'mean-decay' ? 'mean-decay' : init}`,
    value: row.mean,
    color: variant === 'scalar' ? 'var(--c-scalar)' : 'var(--c-channel)',
  }));
  const gapRows: { label: string; block?: Block }[] = [
    { label: 'Gap, default start', block: gaps?.default },
    { label: 'Gap, mean-decay start', block: gaps?.mean_decay },
    { label: 'Gap, joint start', block: gaps?.joint },
  ];

  return (
    <div className="di-results">
      <div className="di-results__bars">
        <AnimatedBars data={bars} max={1} format={(v) => pct(v)} />
      </div>
      <table className="di-table">
        <thead>
          <tr>
            <th>Condition</th>
            {seeds.map((seed) => <th key={seed}>seed {seed}</th>)}
            <th>mean</th>
          </tr>
        </thead>
        <tbody>
          {orderedRows.map(({ init, variant, row }) => (
            <tr key={`${init}/${variant}`}>
              <td>{conditionLabel(init, variant)}</td>
              {row.seeds.map((value, index) => <td key={index}>{pct(value)}</td>)}
              <td><strong>{pct(row.mean)}</strong></td>
            </tr>
          ))}
          {gapRows.filter((item) => item.block).map(({ label, block }) => (
            <tr className="di-table__gap" key={label}>
              <td>{label}</td>
              {block!.seeds.map((value, index) => <td key={index}>{pointsSigned(value)}</td>)}
              <td>{pointsSigned(block!.mean)}</td>
            </tr>
          ))}
        </tbody>
      </table>
      <WhatThisShows>{caption}</WhatThisShows>
    </div>
  );
}

export function ResultsBody({ summary }: { summary: Summary }) {
  const ow = summary.families.overwrite;
  const mq = summary.families.mqar;
  const seeds = summary.seeds;
  return (
    <Panels
      label="Results"
      items={[
        {
          label: 'Learning speed',
          content: (
            <MeasureView
              rows={ow.measures.learning ?? []}
              gaps={ow.contrasts.learning}
              seeds={seeds}
              caption="Overwrite: average unrelated-fact retention during training. The joint series comes from its separate three-seed study."
            />
          ),
        },
        {
          label: 'Fresh examples',
          content: (
            <MeasureView
              rows={ow.measures.fresh ?? []}
              gaps={ow.contrasts.fresh}
              seeds={seeds}
              caption="Overwrite: final accuracy on fresh examples. Joint was not included in this confirmation run."
            />
          ),
        },
        {
          label: 'Same comparison, MQAR',
          content: (
            <MeasureView
              rows={mq.measures.learning ?? []}
              gaps={mq.contrasts.learning}
              seeds={seeds}
              caption="MQAR: average query accuracy during training. Joint values come from its separate three-seed study."
            />
          ),
        },
      ]}
    />
  );
}

export function ResultsVisual() {
  const summary = useDecaySummary();
  return (
    <Remote state={summary} what="the saved results" tall>
      {(data) => <ResultsBody summary={data} />}
    </Remote>
  );
}

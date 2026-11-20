import { AnimatedBars } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote, conditionLabel, pct, pointsSigned } from './common';
import { useDecaySummary } from './data';
import type { Block, MeasureRow, Summary } from './types';

const ORDER: { init: 'default' | 'mean-decay'; variant: 'scalar' | 'channel-wise' }[] = [
  { init: 'default', variant: 'scalar' },
  { init: 'default', variant: 'channel-wise' },
  { init: 'mean-decay', variant: 'scalar' },
  { init: 'mean-decay', variant: 'channel-wise' },
];

/** One saved measure: four conditions as bars (mean of three seeds), then each seed and the paired gaps. */
function MeasureView({
  rows,
  gaps,
  seeds,
  caption,
}: {
  rows: MeasureRow[];
  gaps: { default: Block; mean_decay: Block; interaction: Block } | undefined;
  seeds: number[];
  caption: string;
}) {
  const bars = ORDER.map((o) => {
    const r = rows.find((x) => x.init === o.init && x.variant === o.variant);
    return {
      label: `${o.variant === 'scalar' ? 'Scalar' : 'Channel'}, ${o.init}`,
      value: r ? r.mean : 0,
      color: o.variant === 'scalar' ? 'var(--c-scalar)' : 'var(--c-channel)',
    };
  });
  return (
    <div className="di-results">
      <div className="di-results__bars">
        <AnimatedBars data={bars} max={1} format={(v) => pct(v)} />
      </div>
      <table className="di-table">
        <thead>
          <tr>
            <th>Condition</th>
            {seeds.map((s) => (
              <th key={s}>seed {s}</th>
            ))}
            <th>mean</th>
          </tr>
        </thead>
        <tbody>
          {ORDER.map((o) => {
            const r = rows.find((x) => x.init === o.init && x.variant === o.variant);
            if (!r) return null;
            return (
              <tr key={`${o.init}/${o.variant}`}>
                <td>{conditionLabel(o.init, o.variant)}</td>
                {r.seeds.map((v, i) => (
                  <td key={i}>{pct(v)}</td>
                ))}
                <td>
                  <strong>{pct(r.mean)}</strong>
                </td>
              </tr>
            );
          })}
          {gaps && (
            <>
              <tr className="di-table__gap">
                <td>Gap, default start</td>
                {gaps.default.seeds.map((v, i) => (
                  <td key={i}>{pointsSigned(v)}</td>
                ))}
                <td>{pointsSigned(gaps.default.mean)}</td>
              </tr>
              <tr className="di-table__gap">
                <td>Gap, mean-decay start</td>
                {gaps.mean_decay.seeds.map((v, i) => (
                  <td key={i}>{pointsSigned(v)}</td>
                ))}
                <td>{pointsSigned(gaps.mean_decay.mean)}</td>
              </tr>
            </>
          )}
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
              caption="Overwrite task: average unrelated-association retention over training. Higher is faster. Gap = channel-wise minus scalar."
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
              caption="Overwrite task: overall accuracy on fresh examples at the end of training. Gap = channel-wise minus scalar."
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
              caption="MQAR task: average query accuracy over training. Final accuracy is near 100%. Gap = channel-wise minus scalar."
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
      {(s) => <ResultsBody summary={s} />}
    </Remote>
  );
}

import { AnimatedLines, type LineSeries } from '../../components/AnimatedLines';
import { Panels, type PanelItem } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { findRun, hardwareLabel } from './common';
import type { CostSummary, Family, Hardware, Mode } from './types';

const COLORS = { scalar: 'var(--c-scalar)', 'channel-wise': 'var(--c-channel)' } as const;

/** Every saved optimizer-step time for one family, hardware and mode, as two lines (scalar and channel-wise). */
function traces(summary: CostSummary, hardware: Hardware, family: Family, mode: Mode): LineSeries[] {
  return (['scalar', 'channel-wise'] as const).map((variant) => {
    const run = findRun(summary, hardware, family, mode, variant);
    return {
      name: variant === 'scalar' ? 'Scalar decay' : 'Channel-wise decay',
      ys: run.step_seconds,
      color: COLORS[variant],
    };
  });
}

function TracePanel({
  summary,
  hardware,
  family,
  mode,
  title,
}: {
  summary: CostSummary;
  hardware: Hardware;
  family: Family;
  mode: Mode;
  title: string;
}) {
  const n = findRun(summary, hardware, family, mode, 'scalar').step_seconds.length;
  return (
    <section className="cost-panel cost-panel--tight">
      <h3 className="cost-h3">
        {title} <span className={`cost-hw-inline cost-hw-inline--${hardware}`}>{hardwareLabel(summary, hardware)}</span>
      </h3>
      <AnimatedLines
        series={traces(summary, hardware, family, mode)}
        yLabel="seconds per step"
        xLabel="optimizer step"
        xs={Array.from({ length: n }, (_, i) => i + 1)}
        yMax={1.8}
      />
    </section>
  );
}

/** Inside: each timed step, in order, for the saved runs. The raw traces behind the averages on the results slide. */
export function InsidePanels({ summary }: { summary: CostSummary }) {
  const items: PanelItem[] = [
    {
      label: 'A6000, two workers',
      content: <TracePanel summary={summary} hardware="a6000" family="mqar" mode="parallel" title="MQAR, two workers at once" />,
    },
    {
      label: 'A6000, one after the other',
      content: (
        <TracePanel summary={summary} hardware="a6000" family="mqar" mode="sequential" title="MQAR, one after the other" />
      ),
    },
    {
      label: 'Kaggle T4',
      content: <TracePanel summary={summary} hardware="t4" family="mqar" mode="not recorded" title="MQAR, Kaggle T4 (mode not recorded)" />,
    },
  ];
  return (
    <div className="cost-inside">
      <Panels label="Step traces" items={items} />
      <WhatThisShows>Every timed step in order. The first step is slowest in each run; the rest form a narrow band.</WhatThisShows>
    </div>
  );
}

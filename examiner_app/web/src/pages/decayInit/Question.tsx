import { Remote, pct, pointsSigned } from './common';
import { useDecaySummary } from './data';
import type { MeasureRow, Summary } from './types';

const STARTS: { init: 'default' | 'mean-decay'; label: string }[] = [
  { init: 'default', label: 'Default start' },
  { init: 'mean-decay', label: 'Mean-decay start' },
];
const DESIGNS: { variant: 'scalar' | 'channel-wise'; label: string }[] = [
  { variant: 'scalar', label: 'Scalar decay' },
  { variant: 'channel-wise', label: 'Channel-wise decay' },
];

function cellOf(rows: MeasureRow[], init: string, variant: string): MeasureRow | undefined {
  return rows.find((r) => r.init === init && r.variant === variant);
}

/** Two starts by two decay designs: saved overwrite learning-speed means (mean of 3 seeds), with the gap below. */
function StartByDesignGrid({ summary }: { summary: Summary }) {
  const rows = summary.families.overwrite.measures.learning ?? [];
  const gap = summary.families.overwrite.contrasts.learning;
  return (
    <div className="di-grid" role="table" aria-label="Learning speed by start and decay design">
      <div className="di-grid__corner" role="columnheader">
        Learning speed, mean of 3 seeds
      </div>
      {STARTS.map((s) => (
        <div key={s.init} className="di-grid__head" role="columnheader">
          {s.label}
        </div>
      ))}
      {DESIGNS.map((d) => (
        <div key={d.variant} className="di-grid__row" role="row">
          <div className="di-grid__rowhead" role="rowheader">
            {d.label}
          </div>
          {STARTS.map((s) => {
            const cell = cellOf(rows, s.init, d.variant);
            return (
              <div key={s.init} className="di-grid__cell" role="cell">
                <span className="di-grid__num">{cell ? pct(cell.mean) : '—'}</span>
              </div>
            );
          })}
        </div>
      ))}
      <div className="di-grid__row di-grid__row--gap" role="row">
        <div className="di-grid__rowhead" role="rowheader">
          Channel-wise minus scalar
        </div>
        <div className="di-grid__cell" role="cell">
          <span className="di-grid__num">{gap ? pointsSigned(gap.default.mean) : '—'}</span>
        </div>
        <div className="di-grid__cell" role="cell">
          <span className="di-grid__num">{gap ? pointsSigned(gap.mean_decay.mean) : '—'}</span>
        </div>
      </div>
    </div>
  );
}

export function QuestionVisual() {
  const summary = useDecaySummary();
  return (
    <Remote state={summary} what="the saved results" tall>
      {(s) => <StartByDesignGrid summary={s} />}
    </Remote>
  );
}

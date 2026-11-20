import { Stat } from '../../components/Stat';
import { WhatThisShows } from '../../components/WhatThisShows';
import { pct } from './common';
import type { ControlsSummary } from './data';

/** Section 1: the two outside benchmarks, each with its count of models or runs that show the shortcut. */
export function QuestionVisual({ summary }: { summary: ControlsSummary }) {
  const z = summary.zoology;
  const k = summary.kata;
  const zTotal = z.models_total;
  const kTotal = k.runs_total;
  // Same wording rule as the saved verdict: the share of the flagged model's wrong answers that are position-anchored.
  const z3 = z.verdicts.Z3_anchored_errors as { anchored_share: number | null } | undefined;
  const anchoredLine =
    z3 && z3.anchored_share !== null
      ? `Position-anchored answers: ${pct(z3.anchored_share, 0)} of the flagged model's wrong answers at two and four blank tokens. Not confirmed as position-anchored: a majority would be needed.`
      : null;
  return (
    <div className="ctl-question">
      <div className="ctl-card">
        <span className="ctl-kicker">Outside benchmark 1</span>
        <h3>Standard Zoology recall test</h3>
        <p className="ctl-card__text">Each key is written once: no rewrites.</p>
        <Stat value={zTotal ? z.shortcut_count / zTotal : 0} format={(v) => `${Math.round(v * zTotal)} of ${zTotal}`} label="models that show a position shortcut" />
        {anchoredLine && <p className="ctl-card__note">{anchoredLine}</p>}
      </div>
      <div className="ctl-card">
        <span className="ctl-kicker">Outside benchmark 2</span>
        <h3>KATA overwrite replication</h3>
        <p className="ctl-card__text">Written in passes, then rewritten, like ours.</p>
        <Stat value={kTotal ? k.shortcut_runs / kTotal : 0} format={(v) => `${Math.round(v * kTotal)} of ${kTotal}`} label="runs that show the rewrite shortcut" />
      </div>
      <div className="ctl-question__note">
        <WhatThisShows>Both counts come from saved results. A low count in both benchmarks points the shortcut at our generator, not the architecture.</WhatThisShows>
      </div>
    </div>
  );
}

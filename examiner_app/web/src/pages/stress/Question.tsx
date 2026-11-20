import { AnimatedBars } from '../../components/AnimatedBars';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Stat } from '../../components/Stat';
import { OVERWRITE_IDS, averageOver, pct, settingsFor } from './derive';
import { Remote } from './common';
import type { StressSummary } from './types';
import { useStressSummary } from './data';

const STEP_LABEL: Record<string, string> = {
  baseline: 'Standard test',
  more_keys: 'More keys',
  farther_query: 'Query further away',
  more_rewrites: 'Three rewrites',
};

/** Newest-value recall for each overwrite condition, averaged over the four trained settings. */
function QuestionBody({ s }: { s: StressSummary }) {
  const averages = OVERWRITE_IDS.map((id) => ({
    id,
    value: averageOver(settingsFor(s, 'overwrite', id), (r) => r.latest),
  }));
  const standard = averages[0].value;
  const hardest = averages[averages.length - 1].value;
  return (
    <div className="stress-question">
      <div className="stress-question__stats">
        <Stat value={standard} label="newest value recalled, standard test" format={(v) => pct(v, 0)} />
        <Stat value={hardest} label="newest value recalled, three rewrites" format={(v) => pct(v, 0)} tone="bad" />
      </div>
      <AnimatedBars
        max={1}
        data={averages.map((a) => ({ label: STEP_LABEL[a.id] ?? a.id, value: a.value, color: a.id === 'baseline' ? 'var(--c-good)' : 'var(--accent)' }))}
        format={(v) => pct(v, 0)}
      />
      <WhatThisShows>average of the four trained settings; each step changes one thing from the standard test.</WhatThisShows>
    </div>
  );
}

export function QuestionVisual() {
  const summary = useStressSummary();
  return (
    <Remote state={summary} what="the saved stress results" tall>
      {(s) => <QuestionBody s={s} />}
    </Remote>
  );
}

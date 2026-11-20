import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { pct, pct2 } from './data';
import type { AttentionControlSummary } from './types';

const ATTENTION = 'var(--tok-value)';
const SCALAR = 'var(--c-scalar)';
const CHANNEL = 'var(--c-channel)';
const GUESS = 'var(--c-arm-a0)';

/** Same-budget bars for one test: attention, scalar decay, and the best guess that uses no keys. */
function SameBudgetBars({ summary, family }: { summary: AttentionControlSummary; family: 'mqar' | 'overwrite' }) {
  const runs = summary.same_protocol[family];
  const data: BarDatum[] = [
    { label: 'Attention', value: runs.attention.answer_free.accuracy, color: ATTENTION },
    { label: 'Scalar decay', value: runs.gdn.answer_free.accuracy, color: SCALAR },
    { label: 'Best guess without keys', value: runs.gdn.key_free_baselines.accuracy, color: GUESS },
  ];
  return <AnimatedBars data={data} max={1} format={(v) => pct(v)} />;
}

/** Scalar and channel-wise decay on the older protocol. Attention was not run on it. */
function ScalarVsChannelBars({ summary }: { summary: AttentionControlSummary }) {
  const s = summary.scalar_vs_channel;
  const data: BarDatum[] = [
    { label: 'Multi-key: scalar', value: s.mqar.C1.answer_free.accuracy, color: SCALAR },
    { label: 'Multi-key: channel-wise', value: s.mqar.C3.answer_free.accuracy, color: CHANNEL },
    { label: 'Overwrite: scalar', value: s.overwrite.C1.answer_free.accuracy, color: SCALAR },
    { label: 'Overwrite: channel-wise', value: s.overwrite.C3.answer_free.accuracy, color: CHANNEL },
  ];
  return <AnimatedBars data={data} max={1} format={(v) => pct2(v)} />;
}

/** Three saved-result views, one at a time. */
export function ResultsPanels({ summary }: { summary: AttentionControlSummary }) {
  return (
    <Panels
      label="Saved results"
      items={[
        {
          label: 'Overwrite test',
          content: (
            <section className="ac-panel">
              <h3 className="ac-h3">Overwrite test, same budget: held-out answers correct</h3>
              <SameBudgetBars summary={summary} family="overwrite" />
              <WhatThisShows>attention reads the newest value far more often than scalar decay does.</WhatThisShows>
            </section>
          ),
        },
        {
          label: 'Multi-key recall',
          content: (
            <section className="ac-panel">
              <h3 className="ac-h3">Multi-key recall (MQAR), same budget: held-out answers correct</h3>
              <SameBudgetBars summary={summary} family="mqar" />
              <WhatThisShows>
                MQAR stores several keys with values, then asks for one value back. Both models stay near zero.
              </WhatThisShows>
            </section>
          ),
        },
        {
          label: 'Scalar vs channel-wise',
          content: (
            <section className="ac-panel">
              <h3 className="ac-h3">Scalar vs channel-wise decay, shorter run: held-out answers correct</h3>
              <ScalarVsChannelBars summary={summary} />
              <WhatThisShows>
                both are at the floor with {summary.protocol.kaggle.steps} steps; attention was not run on this setup.
              </WhatThisShows>
            </section>
          ),
        },
      ]}
    />
  );
}

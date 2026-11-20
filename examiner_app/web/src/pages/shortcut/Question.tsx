import { CountUp, Remote, pointsSigned } from './common';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Stat } from '../../components/Stat';
import { MoreDetail } from '../../components/MoreDetail';
import { useEvaluate } from './data';
import { DEFAULT_MODEL, modelLabel } from './types';

/**
 * Section 1: the same live model on two tests. Ordinary layout with one rewrite (training-style) against the
 * block-first layout with three rewrites. Both numbers come from POST /api/shortcut/evaluate on fresh examples.
 */
export function ValidationVsBlockFirst() {
  const base = { ...DEFAULT_MODEL, distance: 8, examples: 64, example_seed: 1, patch: 'none' as const };
  const ordinary = useEvaluate({ ...base, layout: 'general_nonblock', rewrites: 1 });
  const blockFirst = useEvaluate({ ...base, layout: 'block_first', rewrites: 3 });

  return (
    <div className="sc-question">
      <div className="sc-pair">
        <Remote state={ordinary} what="the usual test">
          {(r) => (
            <article className="sc-card sc-card--ok">
              <Stat value={r.accuracy_latest} tone="ok" label="correct on the usual layout" />
            </article>
          )}
        </Remote>
        <Remote state={blockFirst} what="the block-first test">
          {(r) => (
            <article className="sc-card sc-card--bad">
              <Stat value={r.accuracy_latest} tone="bad" label="correct on block-first, 3 rewrites" />
            </article>
          )}
        </Remote>
      </div>
      {ordinary.data && blockFirst.data && (
        <p className="sc-gap">
          Gap: <strong>
            <CountUp
              value={(blockFirst.data.accuracy_latest - ordinary.data.accuracy_latest) * 100}
              format={(v) => `${pointsSigned(v / 100)}`}
            />
          </strong>
        </p>
      )}
      <WhatThisShows>left is the usual layout, right is block-first. Same model, same number of examples.</WhatThisShows>
      <MoreDetail>
        <p>Model: {modelLabel(DEFAULT_MODEL)}. 64 fresh examples per test, scored on the newest value of the queried key.</p>
        <p>The usual layout has one rewrite, with the question 8 tokens after the memory. Block-first has three rewrites, all before the other keys.</p>
      </MoreDetail>
    </div>
  );
}

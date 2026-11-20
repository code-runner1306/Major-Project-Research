import { Term } from '../../components/Term';
import { START_LABEL, VARIANT_LABEL } from './derive';
import { Remote } from './common';
import { useStressModels } from './data';
import type { StartLabel } from './types';
import type { Variant } from '../../api';

const STARTS: StartLabel[] = ['default', 'mean-decay'];
const VARIANTS: Variant[] = ['scalar', 'channel-wise'];

/** Diagram of one memory step, then the twelve local checkpoints as a start x decay grid. */
export function ModelGridVisual() {
  const models = useStressModels();
  return (
    <Remote state={models} what="the local checkpoints" tall>
      {(list) => {
        const has = (init: StartLabel, variant: Variant, seed: number) =>
          list.some((m) => m.init === init && m.variant === variant && m.seed === seed);
        return (
          <div className="stress-models">
            <div className="stress-models__flow" aria-label="one memory step">
              <div className="stress-node">new token</div>
              <div className="stress-arrow" />
              <div className="stress-node is-gate">
                decay gate (α)
                <small>how much old memory fades</small>
              </div>
              <div className="stress-arrow" />
              <div className="stress-node is-gate">
                write gate (β)
                <small>how strongly the new value overwrites</small>
              </div>
              <div className="stress-arrow" />
              <div className="stress-node">recurrent state</div>
            </div>
            <div className="stress-models__grid" role="table" aria-label="local checkpoints">
              <div className="stress-models__corner" role="columnheader" />
              {VARIANTS.map((v) => (
                <div key={v} className="stress-models__head" role="columnheader">
                  {VARIANT_LABEL[v]}
                </div>
              ))}
              {STARTS.map((init) => (
                <div key={init} className="stress-models__row" role="row">
                  <div className="stress-models__start" role="rowheader">
                    {START_LABEL[init]}
                  </div>
                  {VARIANTS.map((v) => (
                    <div key={v} className="stress-models__cell" role="cell">
                      {[42, 43, 44].map((seed) => (
                        <span key={seed} className={has(init, v, seed) ? 'stress-chip is-local' : 'stress-chip'}>
                          seed {seed}
                          {has(init, v, seed) ? ' · live' : ''}
                        </span>
                      ))}
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </div>
        );
      }}
    </Remote>
  );
}

/** Plain-English notes for the Model section, kept out of the visible slide until opened. */
export const MODEL_DETAIL = (
  <>
    <p>
      Every model is the same small <Term term="gated-deltanet" />. Two things vary: the <Term term="decay-initialization" />{' '}
      (default or mean-decay start) and the <Term term="decay-granularity" /> (<Term term="scalar-decay" /> or{' '}
      <Term term="channel-wise-decay" />).
    </p>
    <p>
      A gate intervention edits one gate of a trained model at test time. The weights are not retrained, and the model is
      restored afterwards.
    </p>
  </>
);

import { Unavailable } from '../../components/Unavailable';
import { Remote, conditionLabel } from './common';
import { useDecayGates, useLocalDecayModels } from './data';
import type { DecayInit, Gates, Granularity, LocalModel } from './types';

const STARTS: DecayInit[] = ['default', 'mean-decay'];
const DESIGNS: Granularity[] = ['scalar', 'channel-wise'];
const SEEDS = [42, 43, 44];

/** The four conditions by three seeds, each cell marked local (checkpoint on this machine) or not. */
function ConditionGrid({ models, gates }: { models: LocalModel[]; gates: Gates }) {
  const local = (init: DecayInit, variant: Granularity, seed: number) =>
    models.some((m) => m.init === init && m.variant === variant && m.seed === seed);
  return (
    <div className="di-models">
      {DESIGNS.map((variant) => (
        <div key={variant} className="di-models__row">
          <div className="di-models__design">
            <strong>{variant === 'scalar' ? 'Scalar decay' : 'Channel-wise decay'}</strong>
            <span>{gates.parameters[variant].toLocaleString('en-GB')} parameters</span>
          </div>
          {STARTS.map((init) => (
            <div key={init} className="di-models__cell">
              <span className="di-models__label">{conditionLabel(init, variant)}</span>
              <div className="di-models__seeds">
                {SEEDS.map((seed) => (
                  <span key={seed} className={local(init, variant, seed) ? 'di-badge is-local' : 'di-badge is-unavailable'}>
                    seed {seed}
                    <em>{local(init, variant, seed) ? 'local' : 'not here'}</em>
                  </span>
                ))}
              </div>
            </div>
          ))}
        </div>
      ))}
    </div>
  );
}

/** Plain-language comparison of what is changed before training begins. */
function InitializationRecipes() {
  return (
    <section className="di-recipes" aria-label="Comparison of decay initialization recipes">
      <article className="di-recipes__card">
        <span className="di-recipes__step">Start 1</span>
        <h3>Default</h3>
        <div className="di-recipes__knobs" aria-label="Random offset and random decay rate">
          <span className="is-neutral">offset</span><span className="is-neutral">rate</span>
        </div>
        <p>Keep the model’s usual random starting offset and decay rate. No average-decay target is fitted.</p>
      </article>
      <article className="di-recipes__card">
        <span className="di-recipes__step">Start 2</span>
        <h3>Mean-decay</h3>
        <div className="di-recipes__knobs" aria-label="Fitted offset; original decay rate">
          <span className="is-tuned">offset</span><span className="is-fixed">rate</span>
        </div>
        <p>Adjust the offset until average starting decay matches each layer’s target. Keep the starting rate unchanged.</p>
      </article>
      <article className="di-recipes__card">
        <span className="di-recipes__step">Start 3</span>
        <h3>Joint</h3>
        <div className="di-recipes__knobs" aria-label="Fitted offset and fitted decay rate">
          <span className="is-tuned">offset</span><span className="is-tuned">rate</span>
        </div>
        <p>Adjust both offset and rate to target average log-decay and input sensitivity.</p>
      </article>
      <p className="di-recipes__note">
        These are starting settings: the parameters remain trainable. Joint comes from a separate study and has no local checkpoint in the grid below.
      </p>
    </section>
  );
}

export function ModelsVisual() {
  const gates = useDecayGates();
  const models = useLocalDecayModels();
  return (
    <div className="di-model-visual">
      <InitializationRecipes />
      <Remote state={gates} what="the model list" tall>
      {(g) => (
        <Remote state={models} what="the local models" tall>
          {(list) =>
            list.length === 0 ? (
              <Unavailable reason="No local overwrite checkpoints were found on this machine." />
            ) : (
              <ConditionGrid models={list} gates={g} />
            )
          }
        </Remote>
      )}
      </Remote>
    </div>
  );
}

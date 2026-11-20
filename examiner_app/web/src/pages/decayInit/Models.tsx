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

export function ModelsVisual() {
  const gates = useDecayGates();
  const models = useLocalDecayModels();
  return (
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
  );
}

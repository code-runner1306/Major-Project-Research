import { WhatThisShows } from '../../components/WhatThisShows';
import { designLabel, type ControlsSummary, type ZoologyInit, type Variant } from './data';

const ORDER: { init: ZoologyInit; variant: Variant }[] = [
  { init: 'default', variant: 'scalar' },
  { init: 'default', variant: 'channel-wise' },
  { init: 'mean-decay', variant: 'scalar' },
  { init: 'mean-decay', variant: 'channel-wise' },
  { init: 'published', variant: 'scalar' },
  { init: 'published', variant: 'channel-wise' },
];

/** Section 2: which models each benchmark used. Zoology models are listed from the saved test; KATA has none on disk. */
export function ModelVisual({ summary }: { summary: ControlsSummary }) {
  const z = summary.zoology;
  const k = summary.kata;
  return (
    <div className="ctl-models">
      <div className="ctl-card ctl-card--wide">
        <div className="ctl-card__head">
          <h3>Zoology: standard recall models</h3>
          <span className="ctl-badge is-local">{z.models_local} of {z.models_total} local</span>
        </div>
        <div className="ctl-design-grid">
          {ORDER.map((d) => {
            const members = z.models.filter((m) => m.init === d.init && m.variant === d.variant);
            return (
              <div key={`${d.init}-${d.variant}`} className="ctl-design">
                <span className="ctl-design__name">{designLabel(d.init, d.variant)}</span>
                <div className="ctl-chips">
                  {members.map((m) => (
                    <span key={m.seed} className={`ctl-chip ${m.local ? 'is-local' : 'is-unavailable'}`} title={m.local ? 'checkpoint on this machine' : 'checkpoint not on this machine'}>
                      seed {m.seed}
                      {m.shortcut && <span className="ctl-chip__flag"> shifted-layout drop</span>}
                    </span>
                  ))}
                </div>
              </div>
            );
          })}
        </div>
        <WhatThisShows>Trained on the standard recall task, not on our overwrite task. Same decay designs as the shortcut models.</WhatThisShows>
      </div>
      <div className="ctl-card ctl-card--wide">
        <div className="ctl-card__head">
          <h3>KATA: one architecture, one decay design</h3>
          <span className="ctl-badge is-unavailable">Models unavailable</span>
        </div>
        <p className="ctl-card__text">
          Scalar decay only. No channel-wise version was run, so this benchmark cannot separate the two decay designs.
          {k.parameters && (
            <>
              {' '}
              Ours has {k.parameters.ours.toLocaleString()} parameters; the paper reports {k.parameters.paper?.toLocaleString() ?? 'n/a'}.
            </>
          )}
        </p>
        <WhatThisShows>The trained KATA checkpoints are not on this machine, so only the saved results are shown.</WhatThisShows>
      </div>
    </div>
  );
}

import { ModelBadge, pct, variantText } from './common';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Stat } from '../../components/Stat';
import { Panels } from '../../components/Panels';
import { initLabel, type PerModel } from './types';

/** Section 2: every model in the confirmation, one tab per starting setup, with local or unavailable badges. */
export function ModelGrid({ rows }: { rows: PerModel[] }) {
  const inits = Array.from(new Set(rows.map((r) => r.init)));
  const local = rows.filter((r) => r.local).length;

  const cellsFor = (init: string) => (
    <div className="sc-models__cells">
      {rows
        .filter((r) => r.init === init)
        .map((r) => {
          const hasValues = r.block_first_r3 !== null && r.general_r3 !== null;
          return (
            <article key={`${r.variant}-${r.seed}`} className={r.local ? 'sc-cell' : 'sc-cell is-unavailable'}>
              <div className="sc-cell__head">
                <strong>{variantText(r.variant)}</strong>
                <span>seed {r.seed}</span>
              </div>
              <ModelBadge local={r.local} />
              {hasValues ? (
                <p className="sc-cell__text">
                  Block-first, 3 rewrites: {pct(r.block_first_r3 as number, 0)}. General: {pct(r.general_r3 as number, 0)}.{' '}
                  {r.shortcut ? <span className="sc-tag is-shortcut">shortcut</span> : <span className="sc-muted">no shortcut</span>}
                </p>
              ) : (
                <p className="sc-cell__text sc-muted">No block-first result saved for this model.</p>
              )}
            </article>
          );
        })}
    </div>
  );

  return (
    <div className="sc-models">
      <div className="sc-models__stats">
        <Stat value={rows.length} format={(v) => String(Math.round(v))} label="models tested" />
        <Stat value={local} format={(v) => String(Math.round(v))} label="run on this machine" />
        <Stat value={rows.length - local} format={(v) => String(Math.round(v))} label="saved results only" />
      </div>
      <Panels items={inits.map((init) => ({ label: initLabel(init), content: cellsFor(init) }))} label="Starting setup" />
      <WhatThisShows>each card is one trained model. The badge shows whether this machine can run it.</WhatThisShows>
    </div>
  );
}

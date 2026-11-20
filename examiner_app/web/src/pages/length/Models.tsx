import { Term } from '../../components/Term';
import { LocalBadge, Remote, type RemoteState } from './common';
import { pct, type LengthSummary, type SweepModel } from './types';

const GROUPS: { init: SweepModel['init']; variant: SweepModel['variant'] }[] = [
  { init: 'default', variant: 'scalar' },
  { init: 'default', variant: 'channel-wise' },
  { init: 'mean-decay', variant: 'scalar' },
  { init: 'mean-decay', variant: 'channel-wise' },
];

/** Model visual: the twelve checkpoints as a grid, and the input layout with the filler inserted after the prefix. */
export function LengthModelsVisual({ state }: { state: RemoteState<LengthSummary> }) {
  return (
    <Remote state={state} what="the model list" tall>
      {(s) => {
        const longest = String(s.lengths[s.lengths.length - 1]);
        return (
          <div className="ln-models">
            <div className="ln-grid" role="table" aria-label="Twelve trained models">
              {GROUPS.map((g) => (
                <div className="ln-grid__group" key={`${g.init}-${g.variant}`} role="row">
                  <div className="ln-grid__head" role="rowheader">
                    <span className="ln-grid__init">{g.init}</span>
                    <span className="ln-grid__variant">{g.variant}</span>
                  </div>
                  {s.models
                    .filter((m) => m.init === g.init && m.variant === g.variant)
                    .sort((a, b) => a.seed - b.seed)
                    .map((m) => (
                      <div className="ln-chip" key={`${m.init}-${m.variant}-${m.seed}`} role="cell">
                        <span className="ln-chip__seed">seed {m.seed}</span>
                        <span className="ln-chip__acc">{pct(m.accuracy[longest] ?? 0, 0)} at {Number(longest).toLocaleString()}</span>
                        <LocalBadge local={m.local} />
                      </div>
                    ))}
                </div>
              ))}
            </div>
            <div className="ln-layout" aria-label="How the input is lengthened">
              <div className="ln-layout__row">
                <div className="ln-seg is-write" style={{ flex: s.prefix_tokens }}>
                  {s.prefix_tokens} write tokens
                </div>
                <div className="ln-seg is-filler" style={{ flex: 20 }}>
                  <Term term="filler-token">filler</Term>, 0 to 960
                </div>
                <div className="ln-seg is-rest" style={{ flex: s.training_length - s.prefix_tokens }}>
                  {s.training_length - s.prefix_tokens} tokens: query keys and filler
                </div>
              </div>
              <p className="ln-layout__note">The model is trained on {s.training_length} tokens; the longest test is {Number(longest).toLocaleString()}.</p>
            </div>
          </div>
        );
      }}
    </Remote>
  );
}

import { Stat } from '../../components/Stat';
import { Remote, type RemoteState } from './common';
import { pct, type LengthSummary } from './types';

/** Question visual: average accuracy at each input length, and how many models hold up at the longest length. */
export function LengthQuestionVisual({ state }: { state: RemoteState<LengthSummary> }) {
  return (
    <Remote state={state} what="the saved length sweep" tall>
      {(s) => {
        const mean = (length: number) => s.means.find((m) => m.length === length)?.mean ?? 0;
        const longest = s.lengths[s.lengths.length - 1];
        const holding = s.cells.filter((c) => c.length === longest && c.accuracy >= 0.9).length;
        const total = s.cells.filter((c) => c.length === longest).length;
        const captions: Record<number, string> = {
          [s.training_length]: 'training length',
        };
        return (
          <div className="ln-question">
            <div className="ln-stats">
              {s.lengths.map((length) => (
                <div className="ln-stat-card" key={length}>
                  <Stat
                    value={mean(length)}
                    format={(v) => pct(v)}
                    label={`average accuracy at ${length.toLocaleString()} tokens${captions[length] ? ` (${captions[length]})` : ''}`}
                  />
                </div>
              ))}
              <div className="ln-stat-card is-accent">
                <Stat
                  value={holding}
                  format={(v) => `${Math.round(v)} / ${total}`}
                  label={`models at or above 90% at ${longest.toLocaleString()} tokens`}
                />
              </div>
            </div>
          </div>
        );
      }}
    </Remote>
  );
}

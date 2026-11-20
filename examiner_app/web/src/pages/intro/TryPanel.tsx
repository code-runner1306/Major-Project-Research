import { EXPERIMENTS } from '../../experiments';
import type { ExperimentGroup } from '../../types';
import { Reveal } from './primitives';
import { ToyDeltaRule } from './ToyDeltaRule';

const GROUP_LABEL: Record<ExperimentGroup, string> = {
  start: 'Start',
  experiments: 'Experiment',
  appendix: 'Appendix',
};

/** Links to every other experiment page, plus the toy again with fresh random keys and values. */
export function TryPanel() {
  const cards = EXPERIMENTS.filter((e) => e.id !== 'intro');
  return (
    <div className="intro-try">
      <h4 className="intro-subhead">Where to go next</h4>
      <div className="intro-cards">
        {cards.map((e, i) => (
          <Reveal key={e.id} index={i % 6}>
            <a className="intro-card" href={`#/${e.id}`}>
              <span className="intro-card__group">{GROUP_LABEL[e.group]}</span>
              <strong>{e.title}</strong>
              <span className="intro-card__line">{e.oneLiner}</span>
            </a>
          </Reveal>
        ))}
      </div>

      <h4 className="intro-subhead">Play with the toy again</h4>
      <p className="intro-note">Press New key and value to draw a fresh key and value, then Play to watch the three steps.</p>
      <ToyDeltaRule compact />
    </div>
  );
}

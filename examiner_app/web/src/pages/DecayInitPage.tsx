import type { ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { pointsSigned } from './decayInit/common';
import { useDecaySummary } from './decayInit/data';
import { QuestionVisual } from './decayInit/Question';
import { ModelsVisual } from './decayInit/Models';
import { CODE_STEPS } from './decayInit/Code';
import { ResultsVisual } from './decayInit/Results';
import { InsideVisual } from './decayInit/Inside';
import { TryVisual } from './decayInit/Try';
import type { Summary } from './decayInit/types';
import './decayInit/decayInit.css';

/** Headline: the channel-wise minus scalar learning gap with the mean-decay start, next to the default start's gap. */
function headlineFrom(s: Summary | null): ExperimentDef['headline'] {
  const gap = s?.families.overwrite.contrasts.learning;
  if (!gap) return undefined;
  return {
    value: pointsSigned(gap.mean_decay.mean),
    label: `channel-wise minus scalar with mean-decay start (default start: ${pointsSigned(gap.default.mean)})`,
  };
}

/** Decay initialization page: question, models, code, saved results, gate values, and a live try-it. */
export function DecayInitPage({ def }: PageProps) {
  const summary = useDecaySummary();
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(summary.data) ?? def.headline };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Does the starting forgetting rate change which decay design learns faster?</Takeaway>
          <p>A mean-decay start sets each layer’s average forgetting before training. The default start is random.</p>
          <MoreDetail>
            <p>
              Two <Term term="decay-granularity" /> options are compared: <Term term="scalar-decay" /> (one <Term term="decay-gate" /> value per token)
              and <Term term="channel-wise-decay" /> (one value per key coordinate).
            </p>
            <p>
              The <Term term="decay-initialization" /> is the start: default, mean-decay, or joint; the latter also targets decay sensitivity.
              half-life.
            </p>
            <p>
              Half-life here is a diagnostic of the average decay, not a measured memory lifetime.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <QuestionVisual />
          <WhatThisShows>Mean learning speed on the overwrite task, three seeds, from saved results.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>One small model, two starts, two decay designs, three seeds.</Takeaway>
          <p>The six learning-curve conditions compare three starts—default, mean-decay, and joint—across scalar and channel-wise decay.</p>
          <MoreDetail>
            <p>
              Every model is the same small <Term term="gated-deltanet" />. The channel-wise model has slightly more <Term term="parameter" />s, because
              its decay gate has one value per key coordinate. The parameter counts shown come from the checkpoints.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <ModelsVisual />
          <WhatThisShows>Badges show which of the twelve checkpoints run on this machine.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Three starts, one shared decay formula.</Takeaway>
          <p>Default uses the model’s usual random start. Mean-decay fits the average alpha; joint fits mean log-decay and sensitivity before training.</p>
          <MoreDetail>
            <p>Highlighted lines have a note. The source is read live from the repository.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <CodeWalkthrough steps={CODE_STEPS} />
          <WhatThisShows>Real source code, with notes on the lines that matter.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>Across seeds, neither mean-decay nor joint gave a consistent learning advantage over default.</Takeaway>
          <p>Learning speed averages retention over training. Joint uses a separate study; fresh-example results remain from the default/mean-decay comparison.</p>
          <MoreDetail>
            <p>
              Three seeds only. The per-seed gaps change sign, so the average is not a stable ranking. Results are saved numbers, not new runs.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ResultsVisual />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Mean-decay starts match each layer’s average; training then spreads the channels.</Takeaway>
          <p>The starting averages are set by design. Training then gives the channels different half-lives.</p>
          <MoreDetail>
            <p>
              Values come from the local checkpoints, run on the calibration examples of their seed. The default start is not recorded, so it has no
              start line.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <InsideVisual />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Run one fresh example through a local model and watch its decay.</Takeaway>
          <MoreDetail>
            <p>
              Examples are built fresh on the server and run through the checkpoint on this computer. Each line shows one layer’s decay as a
              half-life in tokens.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <TryVisual />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

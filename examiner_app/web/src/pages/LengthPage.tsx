import type { ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { useLengthSummary, useLocalLengthModels } from './length/data';
import { LengthQuestionVisual } from './length/Question';
import { LengthModelsVisual } from './length/Models';
import { LengthResultsVisual } from './length/Results';
import { LengthInsideVisual } from './length/Inside';
import { LengthTryVisual } from './length/Try';
import { CODE_STEPS } from './length/codeSteps';
import type { LengthSummary } from './length/types';
import './length/length.css';

/** Headline from the saved sweep: how many of the models at the longest length stay at or above 90%. */
function headlineFrom(s: LengthSummary | null): ExperimentDef['headline'] {
  if (!s) return undefined;
  const longest = s.lengths[s.lengths.length - 1];
  const cells = s.cells.filter((c) => c.length === longest);
  const holding = cells.filter((c) => c.accuracy >= 0.9).length;
  return { value: `${holding} / ${cells.length}`, label: `models at or above 90% at ${longest.toLocaleString()} tokens` };
}

/** Longer sequences and filler decay: saved sweep and decay diagnostic, plus live runs of the local MQAR checkpoints. */
export function LengthPage({ def }: PageProps) {
  const summary = useLengthSummary();
  const models = useLocalLengthModels();
  const localModels = models.data ?? [];
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(summary.data) ?? def.headline };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Longer inputs mostly work, but some models fall sharply at 1,024 tokens.</Takeaway>
          <p>Every model is trained on 64-token inputs, then tested on longer ones made by adding filler.</p>
          <MoreDetail>
            <p>
              The filler is the neutral <Term term="filler-token" />. It is added after the first eight tokens, so the
              stored memory has to last through a longer gap before the question.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <LengthQuestionVisual state={summary} />,
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>The same small models, all trained only on 64-token inputs.</Takeaway>
          <p>The filler goes after the first eight tokens, which hold the four key-value writes.</p>
          <MoreDetail>
            <p>
              Each checkpoint is the same small <Term term="gated-deltanet" /> trained in one of four setups, for three{' '}
              <Term term="seed" />s. Badges show which ones run on this machine.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <LengthModelsVisual state={summary} />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Three short pieces of real code.</Takeaway>
          <p>Step through them in order. Highlighted lines have a note.</p>
          <MoreDetail>
            <p>The insertion of filler, how the 64-token input is built, and the decay gate that acts on memory at each step.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <div className="ln-code">
            <CodeWalkthrough steps={CODE_STEPS} />
          </div>
          <WhatThisShows>real source code, with notes on the lines that matter.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>Most models stay accurate at 256 tokens, while several fail at 1,024; neither decay start prevents every failure.</Takeaway>
          <p>Every number here is read from the saved sweep, with checks against the saved files.</p>
          <MoreDetail>
            <p>
              Each cell is a fresh set of examples with four answers each. Each line averages three seeds; dashed lines use
              the mean-decay start. Failures occur with both decay granularities and both starting setups.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <LengthResultsVisual state={summary} />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Each filler token multiplies stored memory by a factor of at most one.</Takeaway>
          <p>The <Term term="decay-gate" /> shrinks memory a little at each filler token. Lines show how much survives, by layer.</p>
          <MoreDetail>
            <p>
              The product counts only the decay gate. The write strength is left out, so a small product does not by
              itself prove that the memory is lost. Live runs use the real checkpoints on 16 examples.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <LengthInsideVisual models={localModels} />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Pick a checkpoint and a filler count, and watch accuracy and memory change.</Takeaway>
          <p>Examples are built on the server and run through the checkpoint live, on this CPU.</p>
          <MoreDetail>
            <p>
              The input length is 64 plus the filler count. Runs above 16 examples are not offered, because a run with
              the longest filler takes several seconds on this CPU.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <LengthTryVisual models={localModels} />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

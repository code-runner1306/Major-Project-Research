import type { CodeStep, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { useApi, type ApiResult } from '../components/useApi';
import type { LayoutFixSummary } from '../api';
import { DimensionCards } from './intro/DimensionCards';
import { RealMemory } from './intro/RealMemory';
import { ToyDeltaRule } from './intro/ToyDeltaRule';
import { TryPanel } from './intro/TryPanel';
import { WhiteboardAnimation } from './intro/WhiteboardAnimation';
import './intro/intro.css';

const CODE_STEPS: CodeStep[] = [
  {
    title: 'The decay gate',
    plain:
      'Turns the model\'s raw signal for each token into the decay gate α, a number between 0 and 1. In scalar decay there is one number per token; in channel-wise decay there is one per memory column.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.decay',
    notes: [
      {
        match: 'A_log',
        note: 'A learned rate for each head, stored on a log scale so that exp() always gives a positive value.',
      },
      {
        match: 'softplus',
        note: 'Turns the raw projection into a positive number, smoothly.',
      },
      {
        match: 'exp(-',
        note: 'exp of a negative number lies between 0 and 1. So α is a forgetting factor: close to 1 keeps the memory, close to 0 forgets almost all of it.',
      },
    ],
  },
  {
    title: 'The memory update',
    plain:
      'The reference loop that updates the memory one token at a time. It is the same fade, read and write described in the Model section.',
    path: 'recipe_model.py',
    symbol: 'serial_rectangular',
    notes: [
      {
        match: 'state=state*alpha',
        note: 'Step 1, fade: every column of the memory is multiplied by its own α.',
      },
      {
        match: 'residual=v',
        note: 'Step 2, read and compare: subtract what the faded memory returns for this key from the wanted value.',
      },
      {
        match: 'state=state+beta',
        note: 'Step 3, write: add the difference, scaled by the write gate β, spread over the key, to the memory.',
      },
    ],
  },
  {
    title: 'The one line that differs',
    plain:
      'The constructor decides how wide the decay gate is. This single line is the only place where the two designs differ.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.__init__',
    notes: [
      {
        match: "self.gate_width=1 if variant=='scalar' else self.head_k",
        note: 'One gate number per token for scalar decay, and one per key column for channel-wise decay. Everything else in the model is shared.',
      },
    ],
  },
];

const CITATION =
  "Published comparison values marked * on the TinyStories page are from Das (TMLR), 'decay design' study; all other results are from this project's own runs.";

function buildSections(api: ApiResult<LayoutFixSummary>): Section[] {
  return [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <p>
            A <Term term="gated-deltanet">Gated DeltaNet</Term> is a small language model that reads text one{' '}
            <Term term="token" /> at a time. It does not keep the whole text. It keeps one fixed-size table, its{' '}
            <Term term="recurrent-state" />, and updates that table after every token.
          </p>
          <p>
            The table never grows. Each of the 4 layers keeps a grid of 256 × 128 numbers, however long the input is.
          </p>
          <p>
            The study asks about one design choice: how the table fades. Each token gets a{' '}
            <Term term="decay-gate">decay gate (α)</Term>, a number between 0 and 1 that shrinks the old contents before
            anything new is written. With <Term term="scalar-decay" /> there is one such number per token, so the whole
            table fades at the same rate. With <Term term="channel-wise-decay" /> there are 128 numbers per token, one per
            column, so each column can fade at its own speed.
          </p>
        </>
      ),
      visual: (
        <>
          <WhiteboardAnimation />
        </>
      ),
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <p>
            The memory is updated in three steps for every token. First it <strong>fades</strong>: the old memory is
            multiplied by the decay gate. Second it <strong>reads</strong>: the model asks what the faded memory currently
            returns for this token&apos;s <Term term="key" />. Third it <strong>writes</strong>: it adds only the difference
            between the wanted <Term term="value" /> and that lookup, scaled by the{' '}
            <Term term="write-gate">write gate (β)</Term>. This is the <Term term="delta-rule" />.
          </p>
          <p>
            The toy below is a 4 × 4 version of exactly that, computed in your browser with no server. Switch between the
            two decay designs to see how one number per column changes the fade.
          </p>
        </>
      ),
      visual: <ToyDeltaRule />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <p>
            The model&apos;s code is in <code>recipe_model.py</code>. The three steps above appear there as a few lines. Walk
            through them below: highlighted lines have a note, and the third step shows the only line that differs between
            the two designs.
          </p>
        </>
      ),
      visual: (
        <>
          <CodeWalkthrough steps={CODE_STEPS} />
          <WhatThisShows>the real source, read from the repository at display time, with notes on the lines that matter.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <p>
            Both designs keep the same memory grid. What changes is the size of the decay gate and so the number of{' '}
            <Term term="parameter">parameters</Term> in the model. The numbers below come from the saved export of the real
            trained checkpoints.
          </p>
          <p>
            Channel-wise decay adds 66,040 parameters to a model of about 1.6 million, an increase of 4.17%.
          </p>
          <p>
            Adding those parameters did not give a reliable recall advantage in these small models. What mattered more was
            the structure of the training data; see the <a href="#/layout-fix">layout fix experiment</a>.
          </p>
        </>
      ),
      visual: <DimensionCards api={api} />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <p>
            Here the trained model runs on one input sequence of 64 tokens. The first two charts show the decay gate over
            time. Scalar decay is a single row; channel-wise decay has one row for each column of the memory.
          </p>
          <p>
            Below that is one memory step from a real trained layer, shown in three stages: before the write, after the
            fade, and after the write. A hand calculation of the same step matches the model&apos;s own calculation to
            floating-point rounding.
          </p>
        </>
      ),
      visual: <RealMemory api={api} />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <p>
            Each experiment page asks one question about these two designs, using the same six steps. Pick the one you
            want to examine next.
          </p>
          <p>
            You can also run the toy again with a new random key and value.
          </p>
        </>
      ),
      visual: <TryPanel />,
    },
  ];
}

/** Introduction page: what the memory is, how the two decay designs differ, and real checkpoint traces. */
export function IntroPage({ def }: PageProps) {
  const api = useApi<LayoutFixSummary>('/api/layout-fix/summary');
  const notice = api.error ? (
    <div className="intro-error" role="alert">
      <strong>The examiner server is not running.</strong>
      <p>
        The real results and checkpoint traces come from the server. Start the server: examiner_app/start.ps1
      </p>
      <p className="intro-error__detail">Reason: {api.error}</p>
    </div>
  ) : undefined;

  return (
    <>
      <ExperimentPage def={def} sections={buildSections(api)} notice={notice} />
      <p className="intro-citation">{CITATION}</p>
    </>
  );
}

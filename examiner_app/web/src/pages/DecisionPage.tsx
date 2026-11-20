import type { CodeStep, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { useDecisionSummary } from './decision/data';
import { InsideVisual, ModelVisual, QuestionVisual, TryUnavailable } from './decision/visuals';
import { ResultsVisual, SavedTable } from './decision/results';
import './decision/decision.css';

// Code steps: every note's match string is a line (or part of one) in the real feasibility script, checked by grep.
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Prompt',
    plain: 'The case, the question and the five options become one text prompt that ends in "Decision:".',
    path: 'decision_feasibility.py',
    symbol: 'prepare_examples',
    notes: [
      {
        match: 'suffix += "\\nDecision:"',
        note: 'The prompt ends with "Decision:". The model is read at that point, after it has seen the options.',
      },
      {
        match: 'target = target / target.sum()',
        note: 'The training target is a probability for each option, rescaled to sum to 1. No single option is marked right.',
      },
    ],
  },
  {
    title: '2 The new head',
    plain: 'The original word-prediction layer is removed and a new five-way head is added.',
    path: 'decision_feasibility.py',
    symbol: 'DecisionPilot',
    notes: [
      {
        match: 'del backbone.output',
        note: 'The model’s word-prediction layer is removed. The rest of the model is kept and trained further.',
      },
      {
        match: 'self.head = nn.Linear(backbone.d_model, n_choices)',
        note: 'A fresh linear layer turns the model state into one score per option.',
      },
      {
        match: 'final = hidden[torch.arange',
        note: 'Only the state at the last prompt token is used for the pick.',
      },
    ],
  },
  {
    title: '3 The one setting that differs',
    plain: 'The decay granularity is set in one line; the rest of the setup is shared.',
    path: 'decision_feasibility.py',
    symbol: 'load_model',
    notes: [
      {
        match: 'GatedDeltaNetLM(PRODUCTION_MODELS["30M"], variant=variant',
        note: 'This is where scalar or channel-wise decay is chosen. Both versions use the same 30M setup.',
      },
    ],
  },
  {
    title: '4 Training',
    plain: 'Each step pushes the predicted spread over the options toward the target spread.',
    path: 'decision_feasibility.py',
    symbol: 'run_variant',
    notes: [
      {
        match: 'loss = -(targets * F.log_softmax',
        note: 'Soft-label cross-entropy: the loss is smallest when the predicted probabilities match the target probabilities.',
      },
    ],
  },
];

/** Decision fine-tuning appendix: saved feasibility results only. No fine-tuned weights were saved, so nothing runs here. */
export function DecisionPage({ def }: PageProps) {
  const summary = useDecisionSummary();

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Does a short extra round of training change which answer a model picks?</Takeaway>
          <p>The model reads a customer-service case and must pick one of five actions.</p>
          <MoreDetail>
            <p>Each training case gives a probability for every option, rather than one fixed right answer.</p>
            <p>This is a small feasibility check. It is not a final result and it does not test the models on the benchmark split.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <QuestionVisual state={summary} />
          <WhatThisShows>one case goes in, and the model chooses one of the five actions.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>The same small 30M model, with a new five-way choice head.</Takeaway>
          <p>Two versions differ only in their <Term term="decay-granularity" />: <Term term="scalar-decay" /> or <Term term="channel-wise-decay" />.</p>
          <MoreDetail>
            <p>
              Both start from a trained <Term term="gated-deltanet" /> checkpoint. The extra round then adjusts the model together with the new head.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <ModelVisual state={summary} />
          <WhatThisShows>each lane is one decay granularity, run through the same steps.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Four steps in the feasibility script, from prompt to loss.</Takeaway>
          <p>Step through them in order. Highlighted lines have a note.</p>
          <MoreDetail>
            <p>The script loads the 30M checkpoints, adds the head, trains for a few steps and writes a report. It never saves the trained weights.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <CodeWalkthrough steps={CODE_STEPS} />
          <WhatThisShows>real source code from the feasibility script, with notes on the lines that matter.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>The extra round trains, but no saved result shows the answers changed.</Takeaway>
          <p>Training loss fell in both saved runs. The saved files never recorded which answer was picked before and after.</p>
          <MoreDetail>
            <p>
              The reports measure how far the predicted spread is from the target spread. They do not measure which option wins, so they cannot
              answer the question.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ResultsVisual state={summary} />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>The model reads the case as text, then a new head scores each action.</Takeaway>
          <p>The prompt is a fixed layout: the case, the question, the options, then the word "Decision".</p>
          <MoreDetail>
            <p>
              A long case is cut at its end so the question and options always fit. The five labels shown are the ones in the saved report.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <InsideVisual state={summary} />
          <WhatThisShows>schematic of the prompt as the script builds it; no example data is saved.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>This machine cannot run the fine-tuned models, so the saved numbers are all there is.</Takeaway>
          <p>The table holds every number from the two saved reports, one row per run and decay granularity.</p>
          <MoreDetail>
            <p>A try-it would need the fine-tuned weights, and the feasibility script never saves them.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <TryUnavailable state={summary} />
          <SavedTable state={summary} />
        </>
      ),
    },
  ];

  return <ExperimentPage def={def} sections={sections} />;
}

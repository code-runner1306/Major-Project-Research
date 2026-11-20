import type { CodeStep, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { Remote } from './controls/common';
import { useControlsSummary, useInside } from './controls/data';
import { QuestionVisual } from './controls/Question';
import { ModelVisual } from './controls/ModelGrid';
import { ResultsVisual } from './controls/Results';
import { InsideVisual } from './controls/Inside';
import { TryVisual } from './controls/Try';
import './controls/controls.css';

// Code steps: every note's match string is a line (or part of one) in the real source, checked by grep.
const CODE_STEPS: CodeStep[] = [
  {
    title: 'Standard recall task',
    plain: 'The standard Zoology generator: four key-value pairs, then questions.',
    path: 'recipe_data.py',
    symbol: 'mqar',
    notes: [
      {
        match: 'memory[:,0::2],memory[:,1::2]=keys,values',
        note: 'Keys go in even positions and values in odd positions, always starting at position 0.',
      },
    ],
  },
  {
    title: 'Our shifted layouts',
    plain: 'Our copy moves the table after a blank prefix and places each question at a chosen gap.',
    path: 'zoology_layout_test.py',
    symbol: 'make_cell',
    notes: [
      {
        match: 'tokens[row, prefix + 2 * i], tokens[row, prefix + 2 * i + 1] = keys[i], values[i]',
        note: 'The table starts after the blank prefix: 0, 1, 2, 4, 8 or 16 tokens.',
      },
      {
        match: 'gaps = rng.choice(space, PAIRS, replace=False, p=power)',
        note: 'Question gaps are drawn with a power law, so near questions are more common, as in the standard task.',
      },
      {
        match: 'position = q0 + 2 * gap',
        note: 'Each question key is placed at its gap after the table.',
      },
      {
        match: 'anchored[row, position] = tokens[row, 2 * i + 1]',
        note: 'The position-anchored answer: the token that would hold the value in the unshifted layout.',
      },
    ],
  },
  {
    title: 'KATA overwrite data',
    plain: 'Every key is written in pass 1 and again in pass 2, then queried. Extra passes test the rewrite rule.',
    path: 'kata_overwrite_data.py',
    symbol: 'generate',
    notes: [
      {
        match: 'writes[row, k, p] = values[p, k]',
        note: 'Each pass writes a fresh value for every key.',
      },
      {
        match: 'if partial_last and p == passes - 1:',
        note: 'For the partial-pass test, the last pass rewrites only half of the keys.',
      },
    ],
  },
  {
    title: 'KATA decay design',
    plain: 'The KATA model is built with the scalar decay design only.',
    path: 'kata_model.py',
    symbol: 'GDNNoGate.__init__',
    notes: [
      {
        match: "super().__init__(d_model, layer_idx, variant='scalar', backend=backend, heads=heads)",
        note: "variant='scalar' is the scalar decay design. No channel-wise KATA run exists.",
      },
    ],
  },
];

export function ControlsPage({ def }: PageProps) {
  const summary = useControlsSummary();
  const inside = useInside();

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Two outside benchmarks test whether the shortcut is ours or general.</Takeaway>
          <p>Zoology is a standard recall test. KATA repeats keys, as our test does.</p>
          <MoreDetail>
            <p>
              A negative control: if neither benchmark shows the shortcut, it comes from our generator and not from
              Gated DeltaNets in general. Zoology writes each key once, so the rewrite shortcut cannot appear there.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <Remote state={summary} what="the saved results">{(s) => <QuestionVisual summary={s} />}</Remote>,
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>Zoology models use our decay designs; KATA used one scalar design.</Takeaway>
          <p>Zoology models were trained on the standard task, not on our overwrite task.</p>
          <MoreDetail>
            <p>
              Zoology: the default and mean-decay runs in scalar and channel-wise decay, plus published models for
              reference. KATA: one architecture, scalar decay, several learning rates.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <Remote state={summary} what="the model list">{(s) => <ModelVisual summary={s} />}</Remote>,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>The standard generator puts keys and values at fixed positions.</Takeaway>
          <p>Our shifted layouts move that table; KATA writes each key in passes.</p>
          <MoreDetail>
            <p>Each step below shows the real source. Click a highlighted line to read its note.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <div className="ctl-code">
          <CodeWalkthrough steps={CODE_STEPS} />
        </div>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>Standard layouts score near 100%; larger shifts lower several models.</Takeaway>
          <p>Each view is a saved result. One blank token drops one model sharply.</p>
          <MoreDetail>
            <p>Lines are averages over seeds. Every number is read from the saved result files, not typed in.</p>
          </MoreDetail>
        </>
      ),
      visual: <Remote state={summary} what="the saved results">{(s) => <ResultsVisual summary={s} />}</Remote>,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Zoology errors mostly pick another key's value; the shortcut picks the first rewrite.</Takeaway>
          <p>Each bar counts the answers that came from one slot of the memory.</p>
          <MoreDetail>
            <p>Generated by server/controls_export.py from the local checkpoints, on the same examples as the saved test.</p>
          </MoreDetail>
        </>
      ),
      visual: <Remote state={inside} what="the slot histograms">{(d) => <InsideVisual inside={d} />}</Remote>,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Run a Zoology checkpoint live on any layout; KATA is results only.</Takeaway>
          <p>Zoology checkpoints run here on the CPU. KATA checkpoints are not on this machine.</p>
          <MoreDetail>
            <p>Using a model's own seed as the example seed gives the saved test's examples, so results can be checked against the saved table.</p>
          </MoreDetail>
        </>
      ),
      visual: <Remote state={summary} what="the try-it controls">{(s) => <TryVisual summary={s} />}</Remote>,
    },
  ];

  return <ExperimentPage def={def} sections={sections} />;
}

import { useState } from 'react';
import type { CodeStep, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { MoreDetail } from '../components/MoreDetail';
import { Takeaway } from '../components/Takeaway';
import { Term } from '../components/Term';
import { Unavailable } from '../components/Unavailable';
import { pct, useProfile, useSummary } from './selectiveReset/data';
import { InsideVisual, Loaded, ModelVisual, QuestionVisual, ResultsVisual, RunTable, TryVisual } from './selectiveReset/visuals';
import './selectiveReset/selective-reset.css';

// Every note's match string is a line (or part of one) in selective_reset_pilot.py, checked by grep.
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 The task',
    plain: 'Each episode writes 16 key-value pairs into two workspaces, then asks questions about them.',
    path: 'selective_reset_pilot.py',
    symbol: 'episodes',
    notes: [
      {
        match: 'target = rng.integers(2, size=batch)',
        note: 'Each episode picks one workspace, red or blue, as the target. The reset will clear it.',
      },
      {
        match: 'groups = np.where(slot < load, target[:, None], 1 - target[:, None])',
        note: 'The first memory-load keys live in the target workspace. The rest live in the other one.',
      },
      {
        match: 'reset_key = rng.integers(load, size=batch)',
        note: 'The cleared key is one of the target workspace keys.',
      },
      {
        match: 'survivor = rng.integers(load, 16, size=batch)',
        note: 'The survivor is a key in the other workspace. A reset should leave its value alone.',
      },
    ],
  },
  {
    title: '2 The reset token',
    plain: 'The only difference between the two versions of an episode is one token at the event.',
    path: 'selective_reset_pilot.py',
    symbol: 'branch',
    notes: [
      {
        match: 'x[row, event] = np.where(reset, 2 + target, 1)',
        note: 'Token 2 or 3 resets the target workspace. Token 1 is a no-op that changes nothing.',
      },
      {
        match: 'x[:, -2] = 4 + spaces[kind, row]',
        note: 'The last two tokens ask the question: which workspace, then which key.',
      },
      {
        match: 'y = np.where(reset & (kind == 0), EMPTY, answers[kind, row])',
        note: 'After a reset, the correct answer for a cleared key is EMPTY. Otherwise the answer is unchanged.',
      },
    ],
  },
  {
    title: '3 The model',
    plain: 'The model is the research code’s stock recurrent design. Nothing reset-specific is added.',
    path: 'selective_reset_pilot.py',
    symbol: 'train',
    notes: [
      {
        match: 'model = build_tiny_denoiser(VOCAB, LENGTH, model_type=family, d_model=32,',
        note: 'Built from the vendored research code. Only the family changes: channel-wise or scalar decay.',
      },
      {
        match: 'sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps, eta_min=0.0001)',
        note: 'The learning rate eases down over the training steps.',
      },
    ],
  },
  {
    title: '4 Scoring',
    plain: 'Each question type is scored on its own. The joint score needs the cleared key and the survivor both right.',
    path: 'selective_reset_pilot.py',
    symbol: 'evaluate',
    notes: [
      {
        match: 'correct.append(pred == y)',
        note: 'Each question type is marked right or wrong for every episode.',
      },
      {
        match: 'int((correct[0] & correct[1]).sum())',
        note: 'The joint score counts episodes where the cleared key and the survivor were both answered correctly.',
      },
    ],
  },
];

/** Results and checkpoint status come from the API; the page's text uses only these computed values. */
export function SelectiveResetPage({ def }: PageProps) {
  const summary = useSummary();
  const live = summary.data?.live ?? false;
  const profile = useProfile(live);
  const [load, setLoad] = useState(8);
  const data = summary.data;

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Can a model learn to clear part of its memory when asked?</Takeaway>
          <p>A reset should wipe one workspace and leave the other alone. This pilot tests whether the model can.</p>
          <MoreDetail>
            <p>
              The memory holds <Term term="key" /> and <Term term="value" /> pairs, split across a red and a blue workspace. Each
              episode ends with a question about one key.
            </p>
            {data && (
              <p>
                A random guess is right about {pct(data.task.guess_rate)} of the time, since a value is one of {data.task.values}.
              </p>
            )}
          </MoreDetail>
        </>
      ),
      visual: <QuestionVisual />,
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>Nothing new in the model: the reset is just one more input token.</Takeaway>
          <p>The model is the same recurrent design used elsewhere in the project. Only the decay style differs between runs.</p>
          <MoreDetail>
            <p>
              Each run is a small <Term term="gated-deltanet" /> with two layers. The two decay styles are{' '}
              <Term term="channel-wise-decay" /> and <Term term="scalar-decay" />.
            </p>
            {data && (
              <p>
                Each run trained for {data.task.steps.toLocaleString()} steps with batches of {data.task.batch}. Training used a mix of
                memory loads, reset and no-op episodes, and all four questions.
              </p>
            )}
          </MoreDetail>
        </>
      ),
      visual: <ModelVisual state={summary} />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>The task, the reset token and the scoring are a few lines each.</Takeaway>
          <p>The pilot file defines the whole experiment. Each step below points at the lines that matter.</p>
          <MoreDetail>
            <p>The file imports the model from the vendored research code and does not change it.</p>
          </MoreDetail>
        </>
      ),
      visual: <CodeWalkthrough steps={CODE_STEPS} />,
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>Models reliably clear the target workspace, but fail to preserve facts in the untouched workspace.</Takeaway>
          <p>No run gets both answers right at any memory load. Kept keys score below a random guess.</p>
          <MoreDetail>
            {data && (
              <>
                <p>
                  Each run was scored at {data.task.memory_loads.length} memory loads on {data.task.length}-token episodes. The highest joint score across all runs
                  and loads is {pct(data.max_joint, 2)}.
                </p>
                <RunTable runs={data.runs} />
              </>
            )}
          </MoreDetail>
        </>
      ),
      visual: (
        <Loaded state={summary} what="the saved results">
          {(s) => <ResultsVisual summary={s} />}
        </Loaded>
      ),
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Except for a rewritten key, the saved model answers EMPTY almost every time.</Takeaway>
          <p>It never returns the older value a key should keep. Its answers for rewritten keys are mostly right.</p>
          <MoreDetail>
            <p>
              This view runs the one saved checkpoint on fixed episodes, so the bars are measured, not inferred. The checkpoint is channel-wise,
              seed 42.
            </p>
          </MoreDetail>
        </>
      ),
      visual: live ? (
        <InsideVisual state={profile} loads={data?.task.memory_loads ?? [1, 2, 4, 8]} load={load} onLoad={setLoad} />
      ) : (
        <Unavailable reason="Models unavailable on this machine. The saved results are on the Results slide." />
      ),
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Build one example, then see what the saved model answers.</Takeaway>
          <p>Pick a memory load, a reset or no-op, and a question. The answer is computed on this machine.</p>
          <MoreDetail>
            <p>
              Each example comes from the pilot's own generator with the example number as its seed. The answer is the checkpoint's top token.
            </p>
          </MoreDetail>
        </>
      ),
      visual: live && data ? (
        <TryVisual summary={data} />
      ) : (
        <Unavailable reason="Models unavailable on this machine." />
      ),
    },
  ];

  return <ExperimentPage def={def} sections={sections} />;
}

import type { ReactNode } from 'react';
import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { Unavailable } from '../components/Unavailable';
import { AnimatedLines } from '../components/AnimatedLines';
import { ExampleError, Skeleton, WaitingForServer } from './dataset/shared';
import { useAttentionSummary, useOverwriteExample } from './attentionControl/data';
import { ExampleStrip } from './attentionControl/ExampleStrip';
import { ModelDiagram } from './attentionControl/ModelDiagram';
import { ResultsPanels } from './attentionControl/ResultsPanels';
import { InsideVisual } from './attentionControl/Inside';
import type { AttentionControlSummary, OverwriteExample } from './attentionControl/types';
import './attentionControl/attentionControl.css';

// Code steps: every match string is copied from attention_control_diagnostic.py and checked by grep.
const CODE_STEPS: CodeStep[] = [
  {
    title: 'The test data',
    plain: 'The same generator makes the data for both models, with fixed settings.',
    path: 'attention_control_diagnostic.py',
    symbol: 'datasets',
    notes: [
      {
        match: 'sequence_lengths=(64 if cpu else 128,)',
        note: 'Sequence length 128 for the real runs. The smoke check uses 64.',
      },
      {
        match: "key_counts=(4,), query_distances=(8,), overwrite_counts=(1,), generator_version='mqar-overwrite-v4'",
        note: 'Four keys, a question eight tokens after the newest write, and one rewrite of the queried key.',
      },
    ],
  },
  {
    title: 'The attention model',
    plain: 'Same backbone; each recurrent memory layer is replaced by causal attention.',
    path: 'attention_control_diagnostic.py',
    symbol: 'AttentionControl',
    notes: [
      {
        match: 'block.attn = CausalSelfAttention(backbone.d_model, backbone.n_head, backbone.d_head)',
        note: 'Every block uses full causal attention in place of the recurrent mixer.',
      },
      {
        match: 'self.positions = nn.Embedding(context, backbone.d_model)',
        note: 'Attention has no built-in sense of order, so each position gets a learned vector.',
      },
    ],
  },
  {
    title: 'The one line that differs',
    plain: 'The recurrent model and the attention model are built on the same line, except for the mixer.',
    path: 'attention_control_diagnostic.py',
    symbol: 'worker',
    notes: [
      {
        match: "construct_model(backbone, 'scalar', config.vocabulary_size,",
        note: 'The recurrent comparison model: scalar decay, built from the same backbone.',
      },
      {
        match: 'model = AttentionControl(backbone, config.vocabulary_size, options, args.seed, config.sequence_lengths[0])',
        note: 'The attention control: same sizes and data, different mixer.',
      },
    ],
  },
  {
    title: 'The verdict rule',
    plain: 'The saved report turns held-out answer accuracy into a cautious finding.',
    path: 'attention_control_diagnostic.py',
    symbol: 'compare',
    notes: [
      {
        match: "a = rows['attention']['confirmation']['answer_free']['exact_query_accuracy']",
        note: 'The verdict uses held-out answers given without the correct earlier answers (answer-free).',
      },
      {
        match: 'elif max(a, g) < .1:',
        note: 'When both models score under 10%, the finding reads: neither establishes learnability.',
      },
    ],
  },
];

/** Headline from the saved overwrite results. */
function headlineFrom(s: AttentionControlSummary | null): ExperimentDef['headline'] {
  return s ? s.headline : undefined;
}

/** Loads a request with the same loading, server-starting and error states as the other pages. */
function Loaded<T>({
  state,
  what,
  children,
}: {
  state: { data: T | null; error: { message: string; status: number | null } | null; waiting: boolean };
  what: string;
  children: (data: T) => ReactNode;
}) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={`loading ${what}`} tall />;
  return <>{children(state.data)}</>;
}

function TryVisual({ summary }: { summary: AttentionControlSummary }) {
  const runs = summary.same_protocol.overwrite;
  const steps = runs.attention.development_curve.map((p) => p.step);
  return (
    <div className="ac-try">
      <Unavailable reason="Models unavailable on this machine: no attention checkpoint is saved here, so there is no live try-it." />
      <section className="ac-panel">
        <AnimatedLines
          series={[
            { name: 'Attention', ys: runs.attention.development_curve.map((p) => p.accuracy), color: 'var(--tok-value)' },
            { name: 'Scalar decay', ys: runs.gdn.development_curve.map((p) => p.accuracy), color: 'var(--c-scalar)' },
          ]}
          xs={steps}
          yLabel="accuracy"
          xLabel="training step"
          yMax={1}
        />
        <WhatThisShows>overwrite test accuracy on development examples, from saved logs checked every 200 steps.</WhatThisShows>
      </section>
    </div>
  );
}

export function AttentionControlPage({ def }: PageProps) {
  const summary = useAttentionSummary();
  const example = useOverwriteExample(0);
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(summary.data) ?? def.headline };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Would a standard attention model pass the same tests as our recurrent models?</Takeaway>
          <p>Attention is the usual alternative: a model that can look back at every earlier token.</p>
          <MoreDetail>
            <p>
              A comparison point helps separate a hard test from a weak memory. The test is the same for both models: a key is written, rewritten,
              and then asked about.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <Loaded state={example} what="the example">
            {(ex: OverwriteExample) => <ExampleStrip example={ex} />}
          </Loaded>
          <WhatThisShows>one real test example: a key is written twice, and the newer value must be returned.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>Same backbone; only the memory mechanism is swapped.</Takeaway>
          <p>The recurrent model keeps one fixed-size table. Attention keeps every earlier token.</p>
          <MoreDetail>
            <p>
              Both models have the same layers, width and feed-forward blocks. The attention model also gets a learned vector for each position, and
              its parameter count differs. This is not a capacity-matched comparison.
            </p>
            <p>
              The recurrent model is the <Term term="gated-deltanet" /> with <Term term="scalar-decay" />.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <Loaded state={summary} what="the model settings">
          {(s) => <ModelDiagram summary={s} />}
        </Loaded>
      ),
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>The test data, the attention swap and the verdict rule sit in one script.</Takeaway>
          <p>Four short steps through the real diagnostic code.</p>
          <MoreDetail>
            <p>Read them in order: the data, the attention model, the one line that differs, and the rule that writes the finding.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <CodeWalkthrough steps={CODE_STEPS} />
          <WhatThisShows>real source code, with notes on the lines that matter.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>Attention solves the overwrite test; the scalar-decay control remains near zero.</Takeaway>
          <p>Bars show held-out answers correct. They are saved results, not new runs.</p>
          <MoreDetail>
            <p>
              The shorter scalar-versus-channel-wise runs use fewer training steps and were not run with attention, so compare them with care.
            </p>
            {summary.data ? (
              <p>
                Outside our tests, the original paper reports channel-wise decay with lower language-modelling loss, by about{' '}
                {Math.abs(summary.data.paper_context.channel_minus_scalar_gap['10M'] ?? 0).toFixed(2)} nats at 10M parameters.
              </p>
            ) : null}
          </MoreDetail>
        </>
      ),
      visual: (
        <Loaded state={summary} what="the saved results">
          {(s) => <ResultsPanels summary={s} />}
        </Loaded>
      ),
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Attention can read every earlier token; recurrent memory keeps one table.</Takeaway>
          <p>Left: the reading pattern of attention. Right: the same tokens folded into one table.</p>
          <MoreDetail>
            <p>The grid uses a real example. Attention weights were not saved with the results, so only what each position may read is shown.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <Loaded state={summary} what="the memory sizes">
          {(s) => (
            <Loaded state={example} what="the example">
              {(ex: OverwriteExample) => <InsideVisual example={ex} summary={s} />}
            </Loaded>
          )}
        </Loaded>
      ),
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>No attention checkpoint is saved here, so there is no live try-it.</Takeaway>
          <p>The training curves below are the saved record of the two overwrite runs.</p>
          <MoreDetail>
            <p>A live try-it needs the attention checkpoint itself. Checkpoints were kept only in Kaggle, so none is stored in this repository.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <Loaded state={summary} what="the training curves">
          {(s) => <TryVisual summary={s} />}
        </Loaded>
      ),
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

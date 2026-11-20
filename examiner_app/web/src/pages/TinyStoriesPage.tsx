import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { MoreDetail } from '../components/MoreDetail';
import { Takeaway } from '../components/Takeaway';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { useLiveModels, useSampleFile, useTsSummary, type TsSummary } from './tinyStories/data';
import { InsideVisual, ModelVisual, QuestionVisual, ResultsVisual, TryVisual } from './tinyStories/visuals';
import './tinyStories/tinyStories.css';

// Every note's match string is a real line in the file (checked with grep when this page was written).
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Stories',
    plain: 'Each story is read from the local validation file, then turned into GPT-2 tokens.',
    path: 'examiner_app/server/dataset_generators.py',
    symbol: '_build_tinystories',
    notes: [
      { match: 'index = seed % len(texts)', note: 'One story per seed, picked from the validation split.' },
      {
        match: 'tokens = (story + [TINYSTORIES_EOT])[:window]',
        note: 'The story is followed by an end-of-text token, then cut to the window length.',
      },
    ],
  },
  {
    title: '2 The model',
    plain: 'Most layers are Gated DeltaNet; a fixed pattern puts ordinary attention in some layers.',
    path: 'model.py',
    symbol: 'GatedDeltaNetBlock',
    notes: [
      {
        match: 'is_full_attention = attention_layout == "hybrid" and layer_idx % 4 == 3',
        note: 'Every fourth layer, counting from the first, uses ordinary attention in the hybrid layout.',
      },
    ],
  },
  {
    title: '3 The line that differs',
    plain: 'The decay gate is one number per head for scalar decay, and one per memory column for channel-wise decay.',
    path: 'model.py',
    symbol: 'GatedDeltaNetAttention',
    notes: [
      {
        match: 'alpha_width = 1 if self.variant == "scalar"',
        note: 'Scalar decay: width 1, so one decay gate per head. Channel-wise: one per column of the head.',
      },
    ],
  },
];

function headlineFrom(s: TsSummary | null): ExperimentDef['headline'] {
  if (!s) return undefined;
  const pairs = s.paired.filter((p) => p.size === '30M');
  const higher = pairs.filter((p) => p.channel_minus_scalar > 0).length;
  return { value: `${higher} / ${pairs.length}`, label: 'paired 30M seeds: channel-wise loss was higher' };
}

/** TinyStories experiment page: question, model, code, results, inside view and a live try-it. */
export function TinyStoriesPage({ def }: PageProps) {
  const summary = useTsSummary();
  const models = useLiveModels();
  const sample = useSampleFile();
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(summary.data) ?? def.headline };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Does the finer fading dial help when models read simple stories?</Takeaway>
          <p>In our runs, no: the single dial had the lower loss at both sizes.</p>
          <MoreDetail>
            <p>
              The models train on short children&rsquo;s stories. The two designs differ only in their{' '}
              <Term term="decay-granularity" />: <Term term="scalar-decay" /> or <Term term="channel-wise-decay" />.
            </p>
            <p>Das et al. report channel-wise decay ahead at every size they tested. Their budget is larger; their numbers are shown starred.</p>
          </MoreDetail>
        </>
      ),
      visual: <QuestionVisual summary={summary} />,
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>Most layers are Gated DeltaNet memory; a few are ordinary attention.</Takeaway>
          <p>Both designs share the same layers. Only the decay gate differs.</p>
          <MoreDetail>
            <p>
              Each Gated DeltaNet layer keeps a fixed-size <Term term="recurrent-state" /> and a <Term term="decay-gate" /> that sets how fast it
              forgets.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ModelVisual summary={summary} />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Three real code steps: the stories, the layers, and the one line that differs.</Takeaway>
          <p>Highlighted lines have a note.</p>
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
          <Takeaway>In these short paired runs, channel-wise decay had slightly higher held-out loss in every seed.</Takeaway>
          <p>The gaps are small: about 0.001 to 0.02 in loss, in the same direction each time.</p>
          <MoreDetail>
            <p>
              Loss is the average surprise at the next token on held-out stories; lower is better. Our 10M and 30M runs used 1,000 steps. The Das
              rows used a larger budget, so compare within a source, not across.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ResultsVisual summary={summary} />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Real decay gates from the checkpoints, while reading one sentence.</Takeaway>
          <p>A value near 1 keeps the memory; near 0 wipes it.</p>
          <MoreDetail>
            <p>
              Each row is one Gated DeltaNet layer; each column is one token. The gates are computed live on CPU from the 30M seed 42 checkpoints.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <InsideVisual sample={sample} summary={summary} />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Type a story opening and compare both decay designs side by side.</Takeaway>
          <p>Each answer comes from the real checkpoint, one token at a time.</p>
          <MoreDetail>
            <p>
              The model picks the most likely next token each time (greedy). Text is tokenised with the GPT-2 tokenizer. Nothing you type is saved.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <TryVisual models={models} sample={sample} />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

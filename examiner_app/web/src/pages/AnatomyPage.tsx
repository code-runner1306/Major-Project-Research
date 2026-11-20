import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { MoreDetail } from '../components/MoreDetail';
import { Remote } from './shortcut/common';
import { Takeaway } from '../components/Takeaway';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { useAnatomySummary } from './anatomy/data';
import { DecayDiagram } from './anatomy/Model';
import { ResultsVisual } from './anatomy/Results';
import { InsideVisual } from './anatomy/Inside';
import { TryVisual } from './anatomy/Try';
import type { Summary } from './anatomy/types';
import './anatomy/anatomy.css';

// Code steps: every note's match string is a line (or part of one) in the real source, checked against the file.
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Test data',
    plain: 'Each test is a table of key–value pairs, a stretch of distractors, then masked queries.',
    path: 'third_party/dualgoose/dualgoose/toy_data.py',
    symbol: 'make_long_retrieval_batch',
    notes: [
      { match: 'table_start = seq_len - tail - gap - 2 * pairs', note: 'With the gap at its maximum, the table starts at position 0.' },
      { match: 'clean = _randint(val_lo, vocab_size - 1, (batch_size, seq_len)', note: 'The whole row starts as random distractor tokens, none of them keys.' },
      { match: 'if collision:', note: 'The collision switch. When it is on, the haystack holds decoy pairs that reuse the table keys.' },
      { match: 'clean[:, gap_start + 2 * d] = key_lo + dk', note: 'Each decoy key is one of the table keys, paired with a wrong value.' },
      { match: 'corrupted[:, apos] = mask_token_id', note: 'Answer slots are replaced by MASK. The model must fill them in.' },
    ],
  },
  {
    title: '2 Model switch',
    plain: 'The model table names each arm. The published cell and the added scalar cell differ in one setting.',
    path: 'third_party/dualgoose/dualgoose/mixers.py',
    symbol: 'BidirectionalRWKVMixer.__init__',
    notes: [
      { match: '"gated_deltanet": {"gated": True},', note: 'The published cell: channel-wise decay. This is our reproduction arm.' },
      { match: '"gated_deltanet_scalar": {"gated": True, "decay": "scalar"},', note: 'Our added arm: the same cell with scalar decay.' },
    ],
  },
  {
    title: '3 The one line that differs',
    plain: 'The decay setting decides how many forgetting values the model predicts per token.',
    path: 'third_party/dualgoose/dualgoose/mixers.py',
    symbol: 'DeltaNetRefLayer',
    notes: [
      { match: 'decay: str = "channel",', note: 'Channel-wise is the default, matching the published code. Scalar is opt-in.' },
      { match: 'self.decay_width = (self.d_key if decay == "channel" else 1) if gated else 0', note: 'One value per key column, or one value shared by all columns.' },
      { match: 'w = torch.sigmoid(parts[4]) if self.gated else None', note: 'The decay gate (α): a value between 0 and 1 for each token.' },
      { match: 'state = state * w[:, ti].unsqueeze(1)', note: 'Before each write the memory is multiplied by the decay. A scalar is shared by every column.' },
    ],
  },
  {
    title: '4 Running a test',
    plain: 'After training, the test scores the masked answers, then scrambles the table values as a control.',
    path: 'third_party/dualgoose/scripts/long_retrieval_probe.py',
    symbol: 'run_trial',
    notes: [
      { match: 'correct += int(pred[m].eq(b.clean[m]).sum().cpu())', note: 'Accuracy: the share of masked answers the model gets exactly right.' },
      { match: 'rc[:, vpos] = b.corrupted[:, vpos.roll(-1)]', note: 'Rebinding control: the table values are shifted in the input, but the answers stay the same.' },
    ],
  },
  {
    title: '5 The statistics',
    plain: 'The comparison between arms is a permutation test on the per-seed accuracies.',
    path: 'third_party/dualgoose/scripts/analyze_granularity.py',
    symbol: 'permutation_p',
    notes: [
      { match: 'observed = abs(mean(a) - mean(b))', note: 'The observed gap between the two arms in average accuracy.' },
      { match: 'hits += abs(mean(x) - mean(y)) >= observed - 1e-12', note: 'Every way of splitting the pooled seeds into two groups is counted. p is the share at least as large.' },
    ],
  },
];

const EXPERIMENT_CARDS: { id: string; blurb: string }[] = [
  { id: 'E1', blurb: 'A table of 32 key–value pairs, then one query whose value is masked.' },
  { id: 'E4', blurb: 'Four pairs, then 48 tokens of distractors, then four masked queries.' },
  { id: 'E5', blurb: 'The same as the haystack test, but the distractors reuse the table keys with wrong values.' },
];

/** Question visual: the three experiments and how many saved runs each arm has. */
function QuestionVisual({ summary }: { summary: ReturnType<typeof useAnatomySummary> }) {
  return (
    <Remote state={summary} what="the saved experiments">
      {(s: Summary) => (
        <div className="anat-cards">
          {EXPERIMENT_CARDS.map((card) => {
            const cell = s.cells.find((c) => c.id === card.id);
            if (!cell) return null;
            const ours = cell.arms.find((a) => a.source === 'ours');
            return (
              <article key={card.id} className="anat-card">
                <h3>{cell.title}</h3>
                <p>{card.blurb}</p>
                <p className="anat-card__meta">
                  {ours ? `${ours.n} training runs per arm` : 'no saved runs'} · locks in at {Math.round(cell.threshold * 100)}%
                </p>
              </article>
            );
          })}
          <WhatThisShows>three saved experiments; each compares channel-wise and scalar decay on the same test.</WhatThisShows>
        </div>
      )}
    </Remote>
  );
}

/** Model visual: real saved channel averages drawn against a schematic scalar bar. */
function ModelVisual({ summary }: { summary: ReturnType<typeof useAnatomySummary> }) {
  return (
    <Remote state={summary} what="the saved decay values">
      {(s: Summary) => <DecayDiagram channelValues={s.channel_decay.E1?.values ?? null} />}
    </Remote>
  );
}

function ResultsView({ summary }: { summary: ReturnType<typeof useAnatomySummary> }) {
  return (
    <Remote state={summary} what="the saved results" tall>
      {(s: Summary) => <ResultsVisual summary={s} />}
    </Remote>
  );
}

function InsideView({ summary }: { summary: ReturnType<typeof useAnatomySummary> }) {
  return (
    <Remote state={summary} what="the saved diagnostics" tall>
      {(s: Summary) => <InsideVisual summary={s} />}
    </Remote>
  );
}

function TryView({ summary }: { summary: ReturnType<typeof useAnatomySummary> }) {
  return (
    <Remote state={summary} what="the saved runs" tall>
      {(s: Summary) => <TryVisual summary={s} />}
    </Remote>
  );
}

/** Anatomy harness page: does channel-wise decay beat scalar decay in the published experiments? */
export function AnatomyPage({ def }: PageProps) {
  const summary = useAnatomySummary();
  const channelAhead = summary.data?.channel_ahead_cells.length ?? null;
  const pageDef: ExperimentDef = {
    ...def,
    headline:
      channelAhead === null
        ? def.headline
        : { value: `${channelAhead} of ${summary.data?.cells.length ?? 3}`, label: 'experiments where channel-wise locks in more runs than scalar' },
  };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Does channel-wise decay beat scalar decay in the published experiments?</Takeaway>
          <p>We re-ran the authors&apos; code as published, then added a scalar-decay version of their model.</p>
          <MoreDetail>
            <p>
              The only difference between the compared arms is <Term term="decay-granularity" />. Our arms use 10 seeds each.
            </p>
            <p>
              A run counts as <Term term="lock-in" /> when it reaches the experiment&apos;s accuracy threshold.
            </p>
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
          <Takeaway>Scalar decay uses one forgetting rate per token; channel-wise gives each column its own.</Takeaway>
          <p>Everything else is identical. Only the forgetting rate changes.</p>
          <MoreDetail>
            <p>
              The published code is <Term term="channel-wise-decay" />, even though the paper&apos;s equation writes a scalar gate. The published
              numbers therefore come from the channel-wise version.
            </p>
            <p>
              The <Term term="decay-gate" /> (α) multiplies the <Term term="recurrent-state" /> before each write. The scalar bars are schematic: one
              shared level, not a measured value.
            </p>
          </MoreDetail>
        </>
      ),
      visual: (
        <>
          <ModelVisual summary={summary} />
          <WhatThisShows>each column is one memory coordinate; its height is the share kept at each step.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>One code switch sets scalar or channel-wise; the rest of the harness is shared.</Takeaway>
          <p>Five steps through the real source, from the test data to the statistics.</p>
          <MoreDetail>
            <p>Highlighted lines have a note. The file paths are relative to the repository root.</p>
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
          <Takeaway>No channel-wise advantage; at 32 pairs, scalar decay was slightly but reliably better.</Takeaway>
          <p>Lock-in counts tie in two experiments. The collision test favours scalar, not significantly.</p>
          <MoreDetail>
            <p>
              Predictions were frozen before the scalar runs. They expected channel-wise to do at least as well in all three. The 32-pair and
              collision results went the other way.
            </p>
            <p>Neither haystack mean gap is significant. Mean accuracy is the average over seeds.</p>
          </MoreDetail>
        </>
      ),
      visual: <ResultsView summary={summary} />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Forcing one shared decay per token cuts recall sharply in trained channel-wise models.</Takeaway>
          <p>Each test is a real generated example. The decay replacements are saved results from trained models.</p>
          <MoreDetail>
            <p>
              The replacements were run on the channel-wise models only. The per-query-position accuracy was not saved, so it is not shown.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <InsideView summary={summary} />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Models are not on this machine, so every saved run is browsable here instead.</Takeaway>
          <p>Filter by experiment or by which runs to show.</p>
          <MoreDetail>
            <p>The rebinding control is the check that a model reads the bindings rather than guessing from the table values.</p>
          </MoreDetail>
        </>
      ),
      visual: <TryView summary={summary} />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

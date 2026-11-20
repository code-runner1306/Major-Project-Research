import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { Takeaway } from '../components/Takeaway';
import { Panels } from '../components/Panels';
import { MoreDetail } from '../components/MoreDetail';
import { Remote } from './shortcut/common';
import { useLocalModels, useShortcutSummary } from './shortcut/data';
import { ValidationVsBlockFirst } from './shortcut/Question';
import { ModelGrid } from './shortcut/ModelGrid';
import { Scoreboard } from './shortcut/Scoreboard';
import { Dumbbell } from './shortcut/Dumbbell';
import { PatchBars } from './shortcut/PatchBars';
import { ErrorMap } from './shortcut/ErrorMap';
import { TryShortcut } from './shortcut/TryShortcut';
import type { ShortcutSummary } from './shortcut/types';
import './shortcut/shortcut.css';

// Code steps: every note's match string is a line (or part of one) in the real source, checked by grep.
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Special layouts',
    plain: 'The named write arrangements, including near misses that change one thing.',
    path: 'confirm_block_layout.py',
    symbol: 'special_patterns',
    notes: [
      {
        match: "'block_first': 'T' * n + 'UUU'",
        note: 'Block-first: all writes of the queried key (T) come first, then three writes of other keys (U). The n writes are the initial value plus the rewrites.',
      },
      {
        match: "'block_after_one_unrelated': 'U' + 'T' * n + 'UU'",
        note: 'Near miss: one unrelated write comes before the run of the queried key.',
      },
      {
        match: "'run_broken_before_latest': 'T' * rewrites + 'U' + 'T' + 'UU'",
        note: 'Near miss: the run is broken by one unrelated write just before the latest write.',
      },
      {
        match: 'if rewrites >= 2:',
        note: 'The initial-then-run layout only exists with two or more rewrites.',
      },
    ],
  },
  {
    title: '2 Fresh layouts',
    plain: 'Fresh examples come from unseen data, sorted into layouts by their write pattern.',
    path: 'confirm_block_layout.py',
    symbol: 'sample_layouts',
    notes: [
      {
        match: 'dataset_seed = seed + SEED_OFFSET + 1000 * cell_index',
        note: 'The seed offset keeps the fresh data apart from the training data.',
      },
      {
        match: 'while any(len(v) < examples for v in chosen.values()):',
        note: 'Keep drawing until every requested layout is full. Rare layouts simply take more draws.',
      },
      {
        match: 'name = layout_of(pattern_of(record), rewrites)',
        note: 'Each example is sorted into a layout by its write pattern.',
      },
    ],
  },
  {
    title: '3 The six rules',
    plain: 'Six candidate rules each give an answer. The model is compared with all six.',
    path: 'confirm_block_layout.py',
    symbol: 'rule_answers',
    notes: [
      {
        match: 'latest = values[-1]',
        note: 'The newest value written for the key. This is the correct answer.',
      },
      {
        match: "opening = pattern.startswith('T' * len(values))",
        note: 'True when the key’s writes form one unbroken run from the first slot, which is the block-first case.',
      },
      {
        match: 'memory_slot_1=memory_values[1]',
        note: 'The value in memory slot 1, the place where the first replacement sits.',
      },
      {
        match: 'opening_run_second=values[1] if opening else latest',
        note: 'The shortcut rule: in a block-first layout it answers the second write. With one rewrite in training, that is the latest value, so the rule was learned without being noticed.',
      },
    ],
  },
  {
    title: '4 Patches',
    plain: 'A patch changes one part of the trained model during a test, then undoes it.',
    path: 'analyze_write_gate.py',
    symbol: 'apply_patch',
    notes: [
      {
        match: "if ctx.patch == 'none' or layer not in ctx.layers:",
        note: 'With no patch the stored keys and write strengths are used exactly as trained.',
      },
      {
        match: 'beta[i, pos] = 1.',
        note: 'Write strength (β) forced to 1 on the latest write, so it overwrites fully.',
      },
      {
        match: 'beta[i, pos] = 0.',
        note: 'Write strength (β) forced to 0 on the first rewrite, so that write is never stored.',
      },
      {
        match: 'k[i, kp], k[i, vp] = k[i, pairs[-1][0]], k[i, pairs[-1][1]]',
        note: 'Every earlier write takes the latest write’s key, so all writes of the key share one address.',
      },
    ],
  },
];

/** Headline from the saved confirmation: how many models follow the shortcut. */
function headlineFrom(s: ShortcutSummary | null): ExperimentDef['headline'] {
  if (!s) return undefined;
  const total = s.per_model.length;
  const withShortcut = s.per_model.filter((m) => m.shortcut).length;
  return { value: `${withShortcut} / ${total}`, label: 'models follow the shortcut at three rewrites' };
}

/** Results visual: three saved-result views, one at a time, each with its own heading. */
function ResultsVisual({ summary }: { summary: ReturnType<typeof useShortcutSummary> }) {
  return (
    <Remote state={summary} what="the saved results" tall>
      {(s) => (
        <div className="sc-results">
          <Panels
            label="Results"
            items={[
              {
                label: 'Hypotheses',
                content: (
                  <section className="sc-panel">
                    <h3 className="sc-h3">Hypotheses, checked against thresholds set in advance</h3>
                    <Scoreboard hypotheses={s.hypotheses} />
                  </section>
                ),
              },
              {
                label: 'Each model',
                content: (
                  <section className="sc-panel">
                    <h3 className="sc-h3">Each model: block-first against general, three rewrites</h3>
                    <Dumbbell rows={s.per_model} />
                  </section>
                ),
              },
              {
                label: 'Discovery patches',
                content: (
                  <section className="sc-panel">
                    <h3 className="sc-h3">Discovery stage: what each patch did</h3>
                    <PatchBars patches={s.patches} />
                  </section>
                ),
              },
            ]}
          />
        </div>
      )}
    </Remote>
  );
}

function ModelsVisual({ summary }: { summary: ReturnType<typeof useShortcutSummary> }) {
  return (
    <Remote state={summary} what="the model list" tall>
      {(s) => <ModelGrid rows={s.per_model} />}
    </Remote>
  );
}

/** Shortcut experiment page: question, models, code, results, an inside story and a try-it. */
export function ShortcutPage({ def }: PageProps) {
  const summary = useShortcutSummary();
  const models = useLocalModels();
  const localModels = models.data ?? [];
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(summary.data) ?? def.headline };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Can a model that looks perfect be using the wrong rule?</Takeaway>
          <p>On the usual test it scores near perfect. Rearranged block-first, the same model falls well short.</p>
          <MoreDetail>
            <p>
              The task is an <Term term="overwrite-task" />: a key is written several times with different <Term term="value" />s, and the model must
              return the newest one.
            </p>
            <p>
              The <Term term="block-first-layout" /> is the same task with every write of the key before the other keys. The gap between the two
              scores is the <Term term="shortcut" />.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ValidationVsBlockFirst />,
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>One small model design, trained in many different ways.</Takeaway>
          <p>Models differ in starting setup, decay granularity and seed. Badges show which ones run on this machine.</p>
          <MoreDetail>
            <p>
              Every model is the same small <Term term="gated-deltanet" />. They differ in their starting <Term term="decay-initialization" />, their{' '}
              <Term term="decay-granularity" /> (<Term term="scalar-decay" /> or <Term term="channel-wise-decay" />), and their <Term term="seed" />.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ModelsVisual summary={summary} />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Four pieces of real code, read live from the repository.</Takeaway>
          <p>Step through them in order. Highlighted lines have a note.</p>
          <MoreDetail>
            <p>
              The layouts, the fresh sampling, the six rules, and the patch used to switch parts of the model off and on, in the order the test uses
              them.
            </p>
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
          <Takeaway>Most tested models show the shortcut at three rewrites.</Takeaway>
          <p>Each hypothesis was set before the fresh data was scored. Every number here comes from saved result files.</p>
          <MoreDetail>
            <p>Three views, in order: the hypotheses, each model on both layouts, and the patches from the discovery stage.</p>
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
          <Takeaway>The model answers from a memory position, not from the key.</Takeaway>
          <p>Each card is one write, in the order it happened. The arrow shows where the answer came from.</p>
          <MoreDetail>
            <p>Each step runs the real local checkpoints live, on fresh examples built with the block-first layout.</p>
            <p>
              Each card is one write, with its key above its value. Cards holding the queried key have a heavy outline and a &ldquo;same key&rdquo; tag.
              The newest of those is tagged &ldquo;newest&rdquo;. The model&apos;s answer is matched to the most recent card holding the same value.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <ErrorMap />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Try any setting and see where the answer came from.</Takeaway>
          <p>Examples are built fresh on the server and run through the model live. A first replacement is the most common wrong answer.</p>
          <MoreDetail>
            <p>
              The <Term term="first-replacement" /> is the most common wrong answer in the block-first layout. The shortcut itself is the rule called{' '}
              <Term term="opening-run-second">the second item of the opening run</Term>.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <TryShortcut models={localModels} />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

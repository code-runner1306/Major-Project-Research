import { useMemo, type ReactNode } from 'react';
import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { MathToggle } from '../components/MathToggle';
import { MoreDetail } from '../components/MoreDetail';
import { Panels } from '../components/Panels';
import { Takeaway } from '../components/Takeaway';
import { Term } from '../components/Term';
import { TokenStrip } from '../components/TokenStrip';
import { WhatThisShows } from '../components/WhatThisShows';
import { useApi } from '../components/useApi';
import type { LayoutFixSummary, ShowcaseExample } from '../api';
import { fmtPct, isBlockFirst, shortcutCounts } from './layoutFix/derive';
import { BlockFirstBars, CurvesPanel, AnswerSources, RerunTable } from './layoutFix/ArmResults';
import { ShortcutDots } from './layoutFix/ShortcutDots';
import { ModelDiagram, ModelSizes } from './layoutFix/ModelDiagram';
import { InsideCompare } from './layoutFix/InsideCompare';
import { TryIt } from './layoutFix/TryIt';
import './layoutFix/layoutFix.css';

// Code steps: line ranges were checked against layout_fix_data.py (ARMS dict at lines 20-27) and
// run_layout_fix_study.py (optimiser and training step at lines 127-150).
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Dataset',
    plain: 'Where training sequences come from: key-value writes, some rewritten, with the latest write placed at random.',
    path: 'retrieval.py',
    symbol: 'RetrievalDataset._overwrite_v4',
    notes: [
      { match: 'rng.sample(value_pool, overwrite_count)', note: 'Picks the new values for the rewrites of the queried key. Each rewrite gets a different value.' },
      { match: 'latest_index = rng.randint', note: 'Chooses where the latest write lands. Later writes in the stream go to other keys or come after it.' },
      { match: 'memory = [token for key, value', note: 'Flattens the writes into the token stream the model reads: key, then value, for each write.' },
    ],
  },
  {
    title: '2 The one line that differs',
    plain: 'The five training-data arms. Each arm changes one or two settings in this table.',
    path: 'layout_fix_data.py',
    lines: { start: 20, end: 27 },
    notes: [
      { match: "'A0_control'", note: 'Control: the original data. The query always comes 8 tokens after the memory, and there is always exactly one rewrite.' },
      { match: "'A1_varied_distance'", note: 'The query distance varies from 8 to 16 tokens, so the answer is not always in the same place.' },
      { match: "'A2_no_block_first'", note: 'Removes block-first examples from the training data, so no example puts every write of the queried key first.' },
      { match: "'A3_mixed_rewrites'", note: 'The number of rewrites varies from 1 to 3, so the latest value is not always the second write.' },
      { match: "'A4_distance_and_rewrites'", note: 'Both changes at once: every combination of distance and rewrite count.' },
    ],
  },
  {
    title: '3 Model',
    plain: "The model's memory update. This is where scalar and channel-wise decay differ.",
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.__init__',
    notes: [
      { match: 'self.gate_width=1', note: 'This is the only scalar/channel-wise difference. Scalar decay has width 1; channel-wise decay has one gate per key column (head_k).' },
      { match: 'self.v_proj', note: 'Projects each token into the value that gets written into memory.' },
      { match: 'self.a_proj', note: 'Produces the decay gate α from each token. Its output width is gate_width.' },
      { match: 'self.b_proj', note: 'Produces the write gate β. A sigmoid keeps it between 0 and 1.' },
    ],
  },
  {
    title: '4 Training',
    plain: 'The training loop, identical for every arm.',
    path: 'run_layout_fix_study.py',
    lines: { start: 127, end: 150 },
    notes: [
      { match: 'CosineAnnealingLR', note: 'The learning rate follows a cosine curve over training, falling smoothly toward zero.' },
      { match: 'for micro in range(offset', note: 'The batch is split into micro-batches of 32. Each adds its share of the gradient, so the result matches one large batch.' },
      { match: 'optimizer.step()', note: 'The only weight update, once per batch of up to 512 examples.' },
    ],
  },
  {
    title: '5 Evaluation: fresh examples',
    plain: 'Fresh examples come from a seed the training data never used.',
    path: 'confirm_block_layout.py',
    symbol: 'sample_layouts',
    notes: [
      { match: 'dataset_seed = seed + SEED_OFFSET', note: 'The fresh data uses its own seed offset, so it does not overlap with training. The overlap audit checks this.' },
      { match: 'while any(len(v) < examples', note: 'Keep drawing until every requested layout is full. Rare layouts simply take more draws.' },
      { match: 'name = layout_of(pattern_of(record)', note: 'Each example is sorted into a layout by its write pattern.' },
    ],
  },
  {
    title: '6 Scoring',
    plain: 'Predictions are compared with the gold answers at each answer position.',
    path: 'confirm_block_layout.py',
    symbol: 'evaluate',
    notes: [
      { match: '.argmax(-1)', note: 'Each prediction is the token with the highest score.' },
      { match: 'outcomes, unrelated = classify(predictions, cell)', note: 'Predictions are sorted into outcomes: latest value right or wrong, and the unrelated-key answers.' },
      { match: 'latest_answers.append(lookup', note: 'The answer at the latest query position is recorded for each example.' },
    ],
  },
];

const RULE_NAME: Record<string, string> = {
  latest: 'the latest write',
  initial: 'the first write',
  second_occurrence: 'the second write of the key',
  memory_slot_1: 'the first memory slot',
  opening_run_second: 'the second item of the opening run',
  final_run_second: 'the second item of the final run',
};

const A4 = 'A4_distance_and_rewrites';

/** Shortcut counts from the API data: all runs, and the A4 runs (the fix). */
function shortcutTotals(s: LayoutFixSummary) {
  const rows = shortcutCounts(s);
  const total = rows.reduce((n, r) => n + r.total, 0);
  const shortcut = rows.reduce((n, r) => n + r.shortcut, 0);
  const a4 = rows.find((r) => r.arm === A4);
  return { total, shortcut, a4Total: a4?.total ?? 0, a4Shortcut: a4?.shortcut ?? 0 };
}

/** Two token strips side by side: a general layout and a block-first layout, with the queried key's writes highlighted. */
function TwoLayouts({ s }: { s: LayoutFixSummary }) {
  const examples: ShowcaseExample[] = [s.showcase_trained_layout, s.showcase];
  const sorted = [...examples].sort((a, b) => Number(isBlockFirst(a.pattern)) - Number(isBlockFirst(b.pattern)));
  const blockFirstEx = sorted.find((e) => isBlockFirst(e.pattern)) ?? sorted[1];
  const pred = blockFirstEx?.predictions['A0_control/scalar'];
  const ruleName = blockFirstEx ? Object.entries(blockFirstEx.rule_answers).find(([, v]) => v === pred)?.[0] : undefined;
  const latestToken = blockFirstEx?.rule_answers.latest;

  return (
    <div className="lf-two">
      {sorted.map((ex, i) => {
        const block = isBlockFirst(ex.pattern);
        return (
          <figure className="lf-panel" key={`${ex.pattern}-${i}`}>
            <figcaption>
              <strong>{block ? 'Block-first layout' : 'General layout'}</strong>
            </figcaption>
            <TokenStrip tokens={ex.tokens} roles={ex.roles} compactFiller />
          </figure>
        );
      })}
      <p className="lf-two__note">
        {blockFirstEx && pred !== undefined && latestToken !== undefined && ruleName && ruleName !== 'latest' ? (
          <>
            The A0 control answered token {pred}, which is {RULE_NAME[ruleName] ?? ruleName}. The newest value is token {latestToken}.
          </>
        ) : (
          'In the block-first layout, every write of the queried key comes first.'
        )}
      </p>
    </div>
  );
}

function buildSections(s: LayoutFixSummary | null): Section[] {
  const skeleton = <div className="lf-skeleton lf-skeleton--tall" aria-label="loading results" />;
  const totals = s ? shortcutTotals(s) : null;
  const ordinary = s ? ordinaryRange(s) : null;
  const ordinaryText = ordinary ? `${fmtPct(ordinary.lo)} to ${fmtPct(ordinary.hi)}` : '…';

  return [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Near-perfect scores can hide a wrong rule.</Takeaway>
          <p>
            On the ordinary test the A0 control scored {ordinaryText}. Rearranging the writes block-first exposes the wrong rule.
          </p>
          <MoreDetail>
            <p>
              The task is an <Term term="overwrite-task" />. The model reads writes: each <Term term="key" /> gets a <Term term="value" />, and
              some keys are written again later (a rewrite). It must return the latest value of one key.
            </p>
            <p>
              In the <Term term="block-first-layout" /> every write of that key comes first. The gap between the two scores is the{' '}
              <Term term="shortcut" />: a position rule that only works for that arrangement.
            </p>
          </MoreDetail>
        </>
      ),
      visual: s ? (
        <>
          <TwoLayouts s={s} />
          <WhatThisShows>the same task in two arrangements. Highlighted squares are the queried key's writes.</WhatThisShows>
        </>
      ) : (
        skeleton
      ),
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>The same small memory model is used in every arm.</Takeaway>
          <p>It keeps a fixed-size memory and updates it one token at a time.</p>
          <MoreDetail>
            <p>
              Each step the <Term term="decay-gate" /> shrinks the old memory, then a <Term term="delta-rule" /> write moves the entry for the
              current key toward the new value, scaled by the <Term term="write-gate" />.
            </p>
            <p>
              Only the training data changes between arms. The two decay designs differ in the width of the decay gate: one number per step
              for <Term term="scalar-decay" />, one per memory column for <Term term="channel-wise-decay" />.
            </p>
            <MathToggle>
              <p>
                Per token t, with key k, value v, decay gate α and write gate β (scalar decay: α is one number; channel-wise: α has one entry
                per key column):
              </p>
              <pre>
                {`S_t = S_(t-1) · diag(α_t)                 decay: shrink the old memory
S_t = S_t + β_t (v_t − S_t k_t) k_tᵀ     delta rule: move the entry for k_t toward v_t`}
              </pre>
            </MathToggle>
          </MoreDetail>
        </>
      ),
      visual: (
        <Panels
          label="Model views"
          items={[
            { label: 'Diagram', content: <ModelDiagram /> },
            { label: 'Sizes', content: <ModelSizes s={s} /> },
          ]}
        />
      ),
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Follow one example from data generation to training.</Takeaway>
          <p>Highlighted lines have a note. The arms are one small table; that table is the whole intervention.</p>
          <MoreDetail>
            <p>Each step shows real source lines from the repository, in the order the experiment runs them.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <div className="lf-code">
          <CodeWalkthrough steps={CODE_STEPS} />
        </div>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>
            {totals
              ? `${totals.shortcut} of ${totals.total} runs show the shortcut; ${
                  totals.a4Shortcut === 0 ? `none of the ${totals.a4Total} A4 runs do` : `${totals.a4Shortcut} of ${totals.a4Total} A4 runs do`
                }.`
              : 'Saved results are loading.'}
          </Takeaway>
          <p>
            A run shows the <Term term="shortcut" /> when its block-first accuracy is at least 20 points below its general-layout accuracy.
          </p>
          <MoreDetail>
            <p>
              Results are on fresh block-first examples with 3 rewrites. Ordinary validation accuracy ranges from {ordinaryText}, so the ordinary
              test cannot tell the arms apart.
            </p>
            <MathToggle>
              <p>
                A run is flagged when <code>general (3 rewrites) − block-first (3 rewrites) ≥ 0.20</code>, using the mean{' '}
                <Term term="fresh-data" /> latest-value accuracy.
              </p>
            </MathToggle>
          </MoreDetail>
        </>
      ),
      visual: s ? (
        <Panels
          label="Result views"
          items={[
            { label: 'Accuracy', content: <BlockFirstBars s={s} /> },
            { label: 'Which runs', content: <ShortcutDots s={s} /> },
            { label: 'Answer sources', content: <AnswerSources s={s} /> },
            { label: 'Ordinary test', content: <CurvesPanel s={s} /> },
            { label: 'Saved vs rerun', content: <RerunTable s={s} /> },
          ]}
        />
      ) : (
        skeleton
      ),
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Both checkpoints read the same example, token by token.</Takeaway>
          <p>The grid is the model's memory; the bars show its gates at each token.</p>
          <MoreDetail>
            <p>
              Both use scalar decay, seed 42, the block-first layout, 3 rewrites, query distance 8 and example seed 7. Scrub once to move both
              panels together.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <InsideCompare />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Change any setting and see which write the model copies.</Takeaway>
          <p>Each answer comes from a live run of the saved checkpoint on the server.</p>
          <MoreDetail>
            <p>
              Start with the A0 preset: block-first with three rewrites. Then switch to A4 to see the same example answered from the latest
              write.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <TryIt />,
    },
  ];
}

/** Final ordinary-validation accuracy of the A0 control runs (the 'it looks solved' number). */
function ordinaryRange(s: LayoutFixSummary): { lo: number; hi: number } | null {
  const finals = s.curves.filter((c) => c.arm === 'A0_control').map((c) => c.final_dev);
  if (finals.length === 0) return null;
  return { lo: Math.min(...finals), hi: Math.max(...finals) };
}

/** Layout fix A0–A4: training-data arms, shortcut evidence, and a live try-it on the real checkpoints. */
export function LayoutFixPage({ def }: PageProps) {
  const api = useApi<LayoutFixSummary>('/api/layout-fix/summary');
  const s = api.data;
  const sections = useMemo(() => buildSections(s), [s]);

  // Hero: the one-liner, analogy and headline come from the saved results (no typed numbers).
  const ordinary = s ? ordinaryRange(s) : null;
  const pageDef: ExperimentDef = {
    ...def,
    oneLiner: ordinary
      ? `Models that score ${fmtPct(ordinary.lo)} to ${fmtPct(ordinary.hi)} on ordinary tests can still learn the wrong rule.`
      : 'Models that score near 100% on ordinary tests can still learn the wrong rule.',
    analogy: 'Like a student who copies the first attempt after an answer is rewritten. Changing the practice fixes the habit.',
    headline: s
      ? {
          value: `${shortcutTotals(s).a4Total - shortcutTotals(s).a4Shortcut} / ${shortcutTotals(s).a4Total}`,
          label: def.headline?.label ?? '',
        }
      : def.headline,
  };

  const notice: ReactNode = api.error ? (
    <div className="lf-error" role="alert">
      <strong>The results could not be loaded.</strong>
      <p>{api.error}</p>
      <p className="lf-muted">Start the server: examiner_app/start.ps1</p>
    </div>
  ) : undefined;
  return <ExperimentPage def={pageDef} sections={sections} notice={notice} />;
}

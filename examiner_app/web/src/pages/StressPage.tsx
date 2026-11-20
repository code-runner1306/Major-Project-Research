import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { QuestionVisual } from './stress/Question';
import { ModelGridVisual, MODEL_DETAIL } from './stress/ModelGrid';
import { ResultsVisual } from './stress/Results';
import { InsideVisual } from './stress/Inside';
import { TryVisual } from './stress/TryStress';
import { useStressSummary } from './stress/data';
import { pct, settingsFor } from './stress/derive';
import type { StressSummary } from './stress/types';
import './stress/stress.css';

// Code steps: every note's match string is a line (or part of one) in the real source, checked by grep.
const CODE_STEPS: CodeStep[] = [
  {
    title: 'Harder cases',
    plain: 'The stress cells: more keys, a query further away, and more rewrites.',
    path: 'stress_recipe_checkpoints.py',
    lines: { start: 18, end: 19 },
    notes: [
      {
        match: "('farther_query',4,24,1)",
        note: 'Query 24 tokens after the write instead of 8. Everything else stays at the standard test.',
      },
      {
        match: "('more_rewrites',4,8,3)",
        note: 'Three rewrites of the key instead of one. The newest value must win over the earlier ones.',
      },
    ],
  },
  {
    title: 'Write-gate patches',
    plain: 'A patch changes one part of the trained model during a test, then undoes it.',
    path: 'analyze_write_gate.py',
    symbol: 'apply_patch',
    notes: [
      {
        match: "if ctx.patch == 'none' or layer not in ctx.layers:",
        note: 'With no patch the stored keys and write strengths are used exactly as trained.',
      },
      {
        match: "if ctx.patch == 'beta_one_latest':",
        note: 'Write strength (β) forced to 1 on the newest write, so it overwrites fully.',
      },
      {
        match: "elif ctx.patch == 'beta_zero_rewrite1':",
        note: 'Write strength (β) forced to 0 on the first replacement, so that write is never stored.',
      },
      {
        match: 'k[i, kp], k[i, vp] = k[i, pairs[-1][0]], k[i, pairs[-1][1]]',
        note: 'Every earlier write takes the newest write’s key, so all writes of the key share one address.',
      },
    ],
  },
  {
    title: 'One forgetting rate',
    plain: 'The decay gate (α) is replaced by its average over channels, at every token.',
    path: 'ablate_recipe_gates.py',
    symbol: 'shared_mean_gates',
    notes: [
      {
        match: 'alpha=original(x)',
        note: 'The trained decay gate is computed as usual first.',
      },
      {
        match: 'return alpha.mean(-1,keepdim=True).expand_as(alpha)',
        note: 'Every channel gets the same forgetting rate. Weights are unchanged, so the rate is the only thing that moves.',
      },
      {
        match: 'if had_attribute:layer.decay=prior',
        note: 'The original method is put back after the probe, so the model is unchanged.',
      },
    ],
  },
  {
    title: 'Where the gates are read',
    plain: 'The decay and write gates for each token, read inside the recurrence.',
    path: 'analyze_write_gate.py',
    symbol: 'instrumented_forward',
    notes: [
      {
        match: 'alpha = module.decay(x)',
        note: 'The decay gate (α) for every token. This is the line the one-forgetting-rate change replaces.',
      },
      {
        match: 'beta = module.b_proj(x)',
        note: 'The write gate (β) for every token.',
      },
      {
        match: 'k, beta = apply_patch(ctx, layer, k, beta)',
        note: 'Any patch is applied here, before the recurrence reads the keys and write strengths.',
      },
    ],
  },
];

/** Headline: newest-value recall with three rewrites, range over the four trained settings. */
function headlineFrom(s: StressSummary | null): ExperimentDef['headline'] {
  if (!s) return undefined;
  const values = settingsFor(s, 'overwrite', 'more_rewrites').map((r) => r.latest ?? 0);
  const low = Math.min(...values);
  const high = Math.max(...values);
  return { value: `${pct(low, 0)}–${pct(high, 0)}`, label: 'newest-value recall with three rewrites, across four settings' };
}

/** Stress tests and gate interventions: harder tests, saved gate results, and live overrides on the local checkpoints. */
export function StressPage({ def }: PageProps) {
  const summary = useStressSummary();
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(summary.data) ?? def.headline };

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Near-perfect on the usual test; much weaker once tests get harder.</Takeaway>
          <p>Harder tests add more keys, more rewrites or a farther query. Recall drops in each case.</p>
          <MoreDetail>
            <p>
              Newest-value recall means the model returns the most recent value written for a key. Each bar averages the four trained
              settings.
            </p>
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
          <Takeaway>Twelve local models: two starts, two decay granularities, three seeds.</Takeaway>
          <p>A gate change edits a trained model at test time, then undoes it.</p>
          <MoreDetail>{MODEL_DETAIL}</MoreDetail>
        </>
      ),
      visual: <ModelGridVisual />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>The harder cases and the gate overrides, read from the real code.</Takeaway>
          <p>Step through in order. Highlighted lines have a note.</p>
          <MoreDetail>
            <p>
              The try section uses the same patch rules and the same channel average as the research scripts, applied to the local
              checkpoints.
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
          <Takeaway>Harder cases reduce recall; forcing gates on trained models causes large drops for some runs.</Takeaway>
          <p>Every bar comes from a saved result file. Averages are over three seeds.</p>
          <MoreDetail>
            <p>
              Gate overrides are reliance tests on already-trained models. They are not fair comparisons with separately trained models, and
              they do not explain learning speed.
            </p>
            <p>Three seeds show descriptive patterns, not significance.</p>
          </MoreDetail>
        </>
      ),
      visual: <ResultsVisual />,
    },
    {
      id: 'inside',
      title: 'Inside',
      explain: (
        <>
          <Takeaway>Same example, two runs: the gates change, and so can the answer.</Takeaway>
          <p>Each card is one write, in order. Tags show which write each run answered from.</p>
          <MoreDetail>
            <p>
              The example is run live through the checkpoint. The gate curves show the decay gate (α) and write gate (β) per token, and
              the size of the <Term term="recurrent-state" />.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <InsideVisual />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Pick a model and a change, then compare fresh examples before and after.</Takeaway>
          <p>The same examples run twice: once as trained, once with the change.</p>
          <MoreDetail>
            <p>
              Examples come from a seed range no saved study used. A scalar model has one <Term term="decay-gate" /> value per head, so
              averaging over channels changes nothing there.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <TryVisual />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

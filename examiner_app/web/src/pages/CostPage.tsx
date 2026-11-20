import type { CodeStep, ExperimentDef, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Stat } from '../components/Stat';
import { Takeaway } from '../components/Takeaway';
import { MoreDetail } from '../components/MoreDetail';
import { WhatThisShows } from '../components/WhatThisShows';
import { Term } from '../components/Term';
import { fmtInt, fmtSigned, Remote } from './cost/common';
import { useCostParameters, useCostSummary } from './cost/data';
import { GateVisual } from './cost/GateVisual';
import { ResultsPanels, stepGapPercent } from './cost/Results';
import { InsidePanels } from './cost/Inside';
import { TryCost } from './cost/Try';
import type { CostSummary, ParametersResponse } from './cost/types';
import './cost/cost.css';

// Code steps: each note's match string is a line (or part of one) in recipe_model.py, checked against the file.
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 The gate width',
    plain: 'One constructor line decides how many gate values each layer produces per token.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN',
    notes: [
      {
        match: "self.gate_width=1 if variant=='scalar' else self.head_k",
        note: 'Scalar decay: one gate value per head. Channel-wise decay: one per key coordinate, so the width is the key size.',
      },
      {
        match: 'self.a_proj=nn.Linear(d_model,heads*self.gate_width,bias=False)',
        note: 'The gate projection maps the layer input to those values, so its size grows with the gate width.',
      },
      {
        match: 'self.A_log=nn.Parameter(rate.log().expand(heads,self.gate_width).clone())',
        note: 'The base timescale is stored once per gate value, so it also grows with the width.',
      },
    ],
  },
  {
    title: '2 How the gate is computed',
    plain: 'The projection output becomes a forgetting factor between 0 and 1 for each gate value.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.decay',
    notes: [
      {
        match: 'raw=self.a_proj(x).view(x.size(0),x.size(1),self.heads,self.gate_width).float()',
        note: 'The projection is reshaped to one row per gate value, for every token.',
      },
      {
        match: 'return torch.exp(-self.A_log.float().exp()*F.softplus(raw+self.dt_bias.float()))',
        note: 'Each value is exp(minus a positive rate times a softplus), so it lies between 0 and 1.',
      },
    ],
  },
  {
    title: '3 Each token’s update',
    plain: 'The gate multiplies the recurrent state before each write. The state keeps the same shape in both models.',
    path: 'recipe_model.py',
    symbol: 'serial_rectangular',
    notes: [
      {
        match: 'state=state*alpha[:,index].unsqueeze(-2)',
        note: 'The gate broadcasts over the state. Scalar decay repeats one value across the key columns, channel-wise decay gives each column its own value.',
      },
      {
        match: 'state=state+beta[:,index].unsqueeze(-1)*residual@key.transpose(-1,-2)',
        note: 'The write itself is unchanged: the same delta-rule update in both models.',
      },
    ],
  },
];

/** Headline and text numbers come from the live parameter counts and the saved summary. */
function headlineFrom(p: ParametersResponse | null): ExperimentDef['headline'] {
  if (!p) return undefined;
  return { value: `+${p.delta.percent_of_scalar.toFixed(2)}%`, label: 'more parameters for channel-wise decay' };
}

/** Range of the saved step-time gaps (channel-wise minus scalar), in percent. */
function gapRanges(s: CostSummary) {
  const a6000 = s.projection.pairs.map((pair) => stepGapPercent(s, 'a6000', pair.family, pair.mode));
  const t4 = (['mqar', 'overwrite'] as const).map((f) => stepGapPercent(s, 't4', f, 'not recorded'));
  const range = (xs: number[]) => [Math.min(...xs), Math.max(...xs)] as const;
  return { a6000: range(a6000), t4: range(t4) };
}

/** True when the first timed step is the slowest in every saved run. */
function firstStepAlwaysSlowest(s: CostSummary): boolean {
  return s.step_timing.every((r) => r.step_seconds[0] > Math.max(...r.step_seconds.slice(1)));
}

/** How much bigger the channel-wise model is than the scalar one, in percent (unsigned text, two decimals). */
function sizeGapPercent(rows: { variant: string; parameters: number }[]): string {
  const ch = rows.find((r) => r.variant === 'channel-wise');
  const sc = rows.find((r) => r.variant === 'scalar');
  if (!ch || !sc) return '…';
  return `${((ch.parameters / sc.parameters - 1) * 100).toFixed(2)}%`;
}

/** Parameter count of the scalar model in millions, for a label. */
function scalarMillions(rows: { variant: string; parameters: number }[]): string {
  const sc = rows.find((r) => r.variant === 'scalar');
  return sc ? `${(sc.parameters / 1e6).toFixed(1)}M` : '…';
}

function QuestionVisual({ summary, params }: { summary: ReturnType<typeof useCostSummary>; params: ReturnType<typeof useCostParameters> }) {
  return (
    <Remote state={params} what="the parameter counts">
      {(p) => (
        <Remote state={summary} what="the saved timing">
          {(s) => {
            const pairSeconds = s.projection.pairs.find((x) => x.family === 'mqar' && x.mode === 'parallel')?.estimated_pair_seconds ?? 0;
            return (
              <div className="cost-question">
                <div className="cost-stats">
                  <Stat value={p.delta.parameters} label="extra parameters, channel-wise" format={(v) => `+${fmtInt(v)}`} />
                  <Stat value={p.delta.percent_of_scalar} label="more than scalar decay" format={(v) => `+${v.toFixed(2)}%`} />
                  <Stat
                    value={pairSeconds / 3600}
                    label="estimated for both models, MQAR, two workers at once (A6000)"
                    format={(v) => `${v.toFixed(1)} h`}
                  />
                </div>
                <WhatThisShows>Parameter counts are live from the recipe model code; the time is a saved A6000 projection.</WhatThisShows>
              </div>
            );
          }}
        </Remote>
      )}
    </Remote>
  );
}

function ModelVisual({ params }: { params: ReturnType<typeof useCostParameters> }) {
  return (
    <Remote state={params} what="the model gate sizes">
      {(p) => (
        <>
          <GateVisual p={p} />
          <WhatThisShows>Each column is one model’s decay gate per layer; the bars are live parameter counts.</WhatThisShows>
        </>
      )}
    </Remote>
  );
}

function ResultsVisual({ summary }: { summary: ReturnType<typeof useCostSummary> }) {
  return <Remote state={summary} what="the saved timing">{(s) => <ResultsPanels summary={s} />}</Remote>;
}

function InsideVisual({ summary }: { summary: ReturnType<typeof useCostSummary> }) {
  return <Remote state={summary} what="the saved step traces">{(s) => <InsidePanels summary={s} />}</Remote>;
}

function TryVisual({ summary }: { summary: ReturnType<typeof useCostSummary> }) {
  return <Remote state={summary} what="the saved timing">{(s) => <TryCost summary={s} />}</Remote>;
}

/** Cost and runtime: how much time and compute the two decay granularities need to train and to run. */
export function CostPage({ def }: PageProps) {
  const summary = useCostSummary();
  const params = useCostParameters();
  const s = summary.data;
  const p = params.data;
  const gaps = s ? gapRanges(s) : null;
  const gapText = (r: readonly [number, number]) => `${fmtSigned(r[0])} to ${fmtSigned(r[1])}`;
  const pageDef: ExperimentDef = { ...def, headline: headlineFrom(p) ?? def.headline };
  const pctText = p ? `${p.delta.percent_of_scalar.toFixed(2)}%` : '…';
  const scalarGate = p ? p.architectures.scalar.layers[0].gate_width : '…';
  const channelGate = p ? p.architectures['channel-wise'].layers[0].gate_width : '…';
  const runSteps = s ? s.step_timing[0]?.accepted_optimizer_steps : undefined;
  const otherFull = s ? s.other_models.full_research_10m : [];
  const otherTiny = s ? s.other_models.tinystories_10m : [];

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <Takeaway>Channel-wise decay adds {pctText} parameters; its step-time cost depends on the hardware.</Takeaway>
          <p>Both models get the same training budget, so the question is what the extra gate costs. Time comes from saved runs, labelled by hardware.</p>
          <MoreDetail>
            <p>
              Each model is a <Term term="gated-deltanet" /> with a <Term term="decay-granularity" />: <Term term="scalar-decay" /> or{' '}
              <Term term="channel-wise-decay" />. Those are the only settings that differ.
            </p>
            <p>
              The extra parameters depend on the model size. Channel-wise decay is {sizeGapPercent(otherFull)} larger than scalar decay in the{' '}
              {scalarMillions(otherFull)} full-research model, and {sizeGapPercent(otherTiny)} larger in the{' '}
              {scalarMillions(otherTiny)} TinyStories model. The recipe model is {pctText}. The other two are different networks, not the recipe.
            </p>
          </MoreDetail>
        </>
      ),
      visual: <QuestionVisual summary={summary} params={params} />,
    },
    {
      id: 'model',
      title: 'Model',
      explain: (
        <>
          <Takeaway>The extra cost sits in one small gate per layer.</Takeaway>
          <p>
            Channel-wise decay widens each layer’s decay gate from {scalarGate} to {channelGate} gate values. The <Term term="recurrent-state" />{' '}
            stays the same size.
          </p>
          <MoreDetail>
            {p && (
              <p>
                Per token, the gate projections do {fmtInt(p.delta.extra_gate_macs_per_token)} more multiply-adds across all layers. That is{' '}
                {(p.delta.extra_gate_macs_share_of_linear * 100).toFixed(2)}% of all linear-layer multiply-adds in the model.
              </p>
            )}
            {p && (
              <p>
                Counts are for the recipe model: width {p.settings.width}, vocabulary {fmtInt(p.settings.vocabulary)}, length {p.settings.length},{' '}
                {p.settings.layers} layers, {p.settings.heads} head, built on CPU with seed {p.settings.seed}.
              </p>
            )}
          </MoreDetail>
        </>
      ),
      visual: <ModelVisual params={params} />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <Takeaway>Only the gate’s width changes; the rest of the model is shared.</Takeaway>
          <p>Read the line that sets the gate width, then how the gate is computed and used at each token.</p>
          <MoreDetail>
            <p>The code is read live from the repository, so every note is on a real line.</p>
          </MoreDetail>
        </>
      ),
      visual: (
        <div className="cost-code">
          <div className="cost-code__box">
            <CodeWalkthrough steps={CODE_STEPS} />
          </div>
          <WhatThisShows>Real source from recipe_model.py, with notes on the lines that change or use the gate.</WhatThisShows>
        </div>
      ),
    },
    {
      id: 'results',
      title: 'Results',
      explain: (
        <>
          <Takeaway>Training speed depends on hardware: the A6000 and T4 comparisons point in different directions.</Takeaway>
          <p>
            {gaps
              ? `Channel-wise minus scalar: ${gapText(gaps.a6000)} on the A6000 and ${gapText(gaps.t4)} on the T4.`
              : 'Loading the saved step times.'}
          </p>
          <MoreDetail>
            <p>
              Each run is one {runSteps ?? '…'}-step window with one seed. Two workers on one GPU slow each step, but finish the pair sooner than
              one after the other. Energy and memory were not recorded for the recipe runs.
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
          <Takeaway>
            {s
              ? firstStepAlwaysSlowest(s)
                ? `In all ${s.step_timing.length} runs the first step is slowest; the rest are steady.`
                : 'The first step is not always the slowest; see the traces.'
              : 'Loading the step traces.'}
          </Takeaway>
          <p>Each line is one saved run, step by step. Hover a line to read its values.</p>
          <MoreDetail>
            <p>The averages on the results slide include the first step, so they sit slightly above the steady rest of each run.</p>
          </MoreDetail>
        </>
      ),
      visual: <InsideVisual summary={summary} />,
    },
    {
      id: 'try',
      title: 'Try it',
      explain: (
        <>
          <Takeaway>Estimate the training time for any number of steps.</Takeaway>
          <p>Pick the hardware, task, mode and variant. The estimate is steps times the measured mean step time for that setting.</p>
          <MoreDetail>
            <p>This is an estimate, not a run. It leaves out setup, validation and checkpoint time, which the saved projection adds back.</p>
          </MoreDetail>
        </>
      ),
      visual: <TryVisual summary={summary} />,
    },
  ];

  return <ExperimentPage def={pageDef} sections={sections} />;
}

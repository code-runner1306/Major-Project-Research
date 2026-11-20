import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Panels, type PanelItem } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { HardwareTag, MODE_LABEL, findRun, fmtBytes, fmtHours, fmtInt, fmtSeconds } from './common';
import type { CostSummary, Family, Mode, StepRun, Variant } from './types';

const MODES: Mode[] = ['parallel', 'sequential'];

/** Channel-wise minus scalar, in percent of the scalar step time. Positive means channel-wise is slower. */
export function stepGapPercent(summary: CostSummary, hardware: 'a6000' | 't4', family: Family, mode: Mode): number {
  const sc = findRun(summary, hardware, family, mode, 'scalar');
  const ch = findRun(summary, hardware, family, mode, 'channel-wise');
  return (ch.mean_step_seconds / sc.mean_step_seconds - 1) * 100;
}

function stepBars(summary: CostSummary, hardware: 'a6000' | 't4', family: Family): BarDatum[] {
  const modes: Mode[] = hardware === 'a6000' ? MODES : ['not recorded'];
  return modes.flatMap((mode) =>
    (['scalar', 'channel-wise'] as Variant[]).map((variant) => {
      const run: StepRun = findRun(summary, hardware, family, mode, variant);
      const label = hardware === 'a6000' ? `${MODE_LABEL[mode]}: ${variant === 'scalar' ? 'scalar' : 'channel-wise'}` : variant;
      return { label, value: run.mean_step_seconds, note: `± ${fmtSeconds(run.se_step_seconds)} (standard error)` };
    }),
  );
}

function StepPanel({ summary, hardware, family }: { summary: CostSummary; hardware: 'a6000' | 't4'; family: Family }) {
  return (
    <section className="cost-panel">
      <h3 className="cost-h3">Seconds per training step: {family === 'mqar' ? 'MQAR' : 'overwrite'} task</h3>
      <HardwareTag summary={summary} hardware={hardware} />
      <AnimatedBars data={stepBars(summary, hardware, family)} format={(v) => fmtSeconds(v, 3)} />
    </section>
  );
}

function ProjectionPanel({ summary }: { summary: CostSummary }) {
  const data: BarDatum[] = summary.projection.pairs.map((p) => ({
    label: `${p.family === 'mqar' ? 'MQAR' : 'Overwrite'}: ${MODE_LABEL[p.mode]}`,
    value: p.estimated_pair_seconds / 3600,
    note: `conservative ${fmtHours(p.conservative_pair_seconds)}`,
  }));
  return (
    <section className="cost-panel">
      <h3 className="cost-h3">
        Both models, {summary.projection.epochs} epochs ({fmtInt(summary.projection.optimizer_steps)} steps each)
      </h3>
      <HardwareTag summary={summary} hardware="a6000" />
      <AnimatedBars data={data} format={(v) => `${v.toFixed(1)} h`} />
    </section>
  );
}

const millions = (n: number) => `${(n / 1e6).toFixed(1)}M`;

function EnergyMemoryPanel({ summary }: { summary: CostSummary }) {
  const fullAll = summary.other_models.full_research_10m;
  const fullSc = fullAll.filter((r) => r.variant === 'scalar');
  const fullCh = fullAll.filter((r) => r.variant === 'channel-wise');
  const mem = fullSc.map((r) => r.peak_allocated_bytes ?? 0);
  const memCh = fullCh.map((r) => r.peak_allocated_bytes ?? 0);
  const tiny = summary.other_models.tinystories_10m;
  const tinySc = tiny.find((r) => r.variant === 'scalar');
  const tinyCh = tiny.find((r) => r.variant === 'channel-wise');
  const sameMem = (xs: number[]) => xs.length > 0 && Math.min(...xs) === Math.max(...xs);
  const energySupported = fullAll.some((r) => r.energy_supported);
  return (
    <section className="cost-panel">
      <h3 className="cost-h3">Energy and memory: what was saved</h3>
      <table className="cost-table">
        <thead>
          <tr>
            <th scope="col">Model and hardware</th>
            <th scope="col">Energy</th>
            <th scope="col">Memory</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <th scope="row">Recipe model, A6000 (timing runs)</th>
            <td>Not recorded</td>
            <td>Not recorded</td>
          </tr>
          <tr>
            <th scope="row">Recipe model, Kaggle T4 (timing runs)</th>
            <td>Not recorded</td>
            <td>Not recorded</td>
          </tr>
          {fullSc[0] && (
            <tr className="cost-table-other">
              <th scope="row">Other model, {millions(fullSc[0].parameters)} parameters, A6000 (full research)</th>
              <td>{energySupported ? 'Measured' : 'Not supported on this A6000 profile'}</td>
              <td>
                {sameMem(mem)
                  ? `${fmtBytes(Math.min(...mem))} peak, scalar (equal in all ${mem.length} runs)`
                  : `${fmtBytes(Math.min(...mem))} to ${fmtBytes(Math.max(...mem))}, scalar`}
                {' · '}
                {sameMem(memCh)
                  ? `${fmtBytes(Math.min(...memCh))} channel-wise`
                  : `${fmtBytes(Math.min(...memCh))} to ${fmtBytes(Math.max(...memCh))} channel-wise`}
              </td>
            </tr>
          )}
          {tinySc && tinyCh && (
            <tr className="cost-table-other">
              <th scope="row">
                Other model, {millions(tinySc.parameters)} parameters, Tesla T4 (seed {tinySc.seed} only)
              </th>
              <td>
                {tinySc.energy_joules !== null && tinyCh.energy_joules !== null
                  ? `scalar ${fmtInt(tinySc.energy_joules)} J · channel-wise ${fmtInt(tinyCh.energy_joules)} J`
                  : 'Not recorded'}
              </td>
              <td>Not recorded</td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="cost-small">
        The recipe runs were not measured for energy or memory. The other-model rows are different networks, shown only to say what
        was measured. They are not comparable with the recipe rows.
      </p>
    </section>
  );
}

/** Results: saved step times per hardware, projected pair times, and what was (and was not) recorded for energy and memory. */
export function ResultsPanels({ summary }: { summary: CostSummary }) {
  const items: PanelItem[] = [
    { label: 'A6000, MQAR', content: <StepPanel summary={summary} hardware="a6000" family="mqar" /> },
    { label: 'A6000, overwrite', content: <StepPanel summary={summary} hardware="a6000" family="overwrite" /> },
    { label: 'Kaggle T4, MQAR', content: <StepPanel summary={summary} hardware="t4" family="mqar" /> },
    { label: 'Kaggle T4, overwrite', content: <StepPanel summary={summary} hardware="t4" family="overwrite" /> },
    { label: 'Projected time', content: <ProjectionPanel summary={summary} /> },
    { label: 'Energy and memory', content: <EnergyMemoryPanel summary={summary} /> },
  ];
  const timed = summary.step_timing[0]?.accepted_optimizer_steps;
  return (
    <div className="cost-results">
      <Panels label="Cost results" items={items} />
      <WhatThisShows>
        Mean seconds per training step over {timed ?? '…'} timed steps. Each panel uses one hardware only.
      </WhatThisShows>
    </div>
  );
}

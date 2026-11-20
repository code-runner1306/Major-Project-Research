import { useState } from 'react';
import { Stat } from '../../components/Stat';
import { WhatThisShows } from '../../components/WhatThisShows';
import { HardwareTag, MODE_LABEL, findRun, fmtHours, fmtInt, hardwareLabel } from './common';
import type { CostSummary, Family, Hardware, Mode, Variant } from './types';

const MAX_STEPS = 100_000; // input limit on the form, not a claim about the data

const MODES_FOR: Record<Hardware, Mode[]> = { a6000: ['parallel', 'sequential'], t4: ['not recorded'] };

/**
 * Live estimate from saved per-step timings: steps x the mean measured step time for the chosen hardware, family, mode and variant.
 * It is an estimate, not a run. Setup, validation and checkpoints are left out (the saved projection adds about 0.3%).
 */
export function TryCost({ summary }: { summary: CostSummary }) {
  const [hardware, setHardware] = useState<Hardware>('a6000');
  const [family, setFamily] = useState<Family>('mqar');
  const [mode, setMode] = useState<Mode>('parallel');
  const [variant, setVariant] = useState<Variant>('scalar');
  const fullSteps = summary.projection.optimizer_steps;
  const [steps, setSteps] = useState<number>(fullSteps);

  const safeSteps = Number.isFinite(steps) ? Math.min(MAX_STEPS, Math.max(1, Math.round(steps))) : fullSteps;
  const run = findRun(summary, hardware, family, mode, variant);
  const estimateSeconds = safeSteps * run.mean_step_seconds;
  const job = hardware === 'a6000' ? summary.projection.jobs.find((j) => j.family === family && j.mode === mode && j.variant === variant) : undefined;

  const changeHardware = (h: Hardware) => {
    setHardware(h);
    setMode(MODES_FOR[h][0]);
  };

  return (
    <div className="cost-try">
      <form className="cost-try-form" onSubmit={(e) => e.preventDefault()} aria-label="Training time estimate">
        <label>
          Hardware
          <select value={hardware} onChange={(e) => changeHardware(e.target.value as Hardware)}>
            <option value="a6000">{hardwareLabel(summary, 'a6000')}</option>
            <option value="t4">{hardwareLabel(summary, 't4')}</option>
          </select>
        </label>
        <label>
          Task
          <select value={family} onChange={(e) => setFamily(e.target.value as Family)}>
            <option value="mqar">MQAR</option>
            <option value="overwrite">Overwrite</option>
          </select>
        </label>
        <label>
          Running mode
          <select value={mode} onChange={(e) => setMode(e.target.value as Mode)}>
            {MODES_FOR[hardware].map((m) => (
              <option key={m} value={m}>
                {MODE_LABEL[m]}
              </option>
            ))}
          </select>
        </label>
        <label>
          Decay granularity
          <select value={variant} onChange={(e) => setVariant(e.target.value as Variant)}>
            <option value="scalar">Scalar decay</option>
            <option value="channel-wise">Channel-wise decay</option>
          </select>
        </label>
        <label>
          Optimizer steps
          <input
            type="number"
            min={1}
            max={MAX_STEPS}
            step={1}
            value={Number.isFinite(steps) ? steps : ''}
            onChange={(e) => setSteps(e.target.valueAsNumber)}
          />
        </label>
      </form>

      <div className="cost-try-out">
        <HardwareTag summary={summary} hardware={hardware} />
        <Stat value={estimateSeconds / 3600} label={`estimated hours for ${fmtInt(safeSteps)} steps (estimate)`} format={(v) => `${v.toFixed(2)} h`} />
        <p className="cost-small">
          {fmtInt(safeSteps)} steps × {run.mean_step_seconds.toFixed(3)} s per step (mean of {run.accepted_optimizer_steps} timed steps) ={' '}
          {fmtHours(estimateSeconds)}. The examples seen are {fmtInt(safeSteps * run.effective_batch)} (batch {run.effective_batch}).
        </p>
        {job && (
          <p className="cost-small">
            Saved projection for the full {fmtInt(summary.projection.optimizer_steps)}-step run: {fmtHours(job.estimated_seconds)} (estimated),{' '}
            {fmtHours(job.conservative_seconds)} (conservative), one model.
          </p>
        )}
        <WhatThisShows>Estimate only: the measured mean step time scaled to the chosen number of steps.</WhatThisShows>
      </div>
    </div>
  );
}

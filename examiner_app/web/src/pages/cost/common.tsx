import type { ReactNode } from 'react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import type { CostSummary, Family, Hardware, Mode, StepRun, Variant } from './types';

export const fmtInt = (v: number) => Math.round(v).toLocaleString('en-US');
export const fmtSeconds = (v: number, digits = 3) => `${v.toFixed(digits)} s`;
export const fmtHours = (seconds: number) => `${(seconds / 3600).toFixed(1)} h`;
export const fmtBytes = (v: number) => `${(v / 1024 ** 3).toFixed(2)} GB`;
export const fmtSigned = (v: number, digits = 1) => `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(digits)}%`;

export const VARIANT_LABEL: Record<Variant, string> = { scalar: 'Scalar decay', 'channel-wise': 'Channel-wise decay' };
export const FAMILY_LABEL: Record<Family, string> = { mqar: 'MQAR', overwrite: 'Overwrite' };
export const MODE_LABEL: Record<Mode, string> = {
  parallel: 'Two workers at once',
  sequential: 'One after the other',
  'not recorded': 'Mode not recorded',
};

/** Hardware name exactly as the saved file records it. Never shortened into a different machine. */
export function hardwareLabel(summary: CostSummary, id: Hardware): string {
  return summary.hardware.find((h) => h.id === id)?.label ?? id;
}

/** Finds one saved run. Throws if it is missing, so a broken export shows an error rather than a blank chart. */
export function findRun(summary: CostSummary, hardware: Hardware, family: Family, mode: Mode, variant: Variant): StepRun {
  const run = summary.step_timing.find(
    (r) => r.hardware === hardware && r.family === family && r.mode === mode && r.variant === variant,
  );
  if (!run) throw new Error(`no saved run for ${hardware} ${family} ${mode} ${variant}`);
  return run;
}

/** Skeleton while loading, retry notice while the server starts, error panel on failure. */
export function Remote<T>({
  state,
  what,
  tall = false,
  children,
}: {
  state: { data: T | null; error: { message: string; status: number | null } | null; waiting: boolean; pending: boolean };
  what: string;
  tall?: boolean;
  children: (data: T) => ReactNode;
}) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={`loading ${what}`} tall={tall} />;
  return <div className={state.pending ? 'cost-remote cost-pending' : 'cost-remote'}>{children(state.data)}</div>;
}

/** Small label that says which hardware a chart uses. Shown on every timing panel. */
export function HardwareTag({ summary, hardware }: { summary: CostSummary; hardware: Hardware }) {
  const kind = summary.hardware.find((h) => h.id === hardware)?.kind ?? 'measured';
  return (
    <p className={`cost-hw cost-hw--${hardware}`}>
      <strong>{hardwareLabel(summary, hardware)}</strong> <span>{kind}</span>
    </p>
  );
}

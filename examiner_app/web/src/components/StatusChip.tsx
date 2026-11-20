import type { ExperimentStatus } from '../types';

const LABELS: Record<ExperimentStatus, string> = {
  live: 'live',
  results: 'results',
  unavailable: 'unavailable',
};

const TIPS: Record<ExperimentStatus, string> = {
  live: 'Checkpoints run on this machine',
  results: 'Saved results only',
  unavailable: 'Not available in this build',
};

export function StatusChip({ status }: { status: ExperimentStatus }) {
  return (
    <span className={`chip chip-${status}`} title={TIPS[status]}>
      {LABELS[status]}
    </span>
  );
}

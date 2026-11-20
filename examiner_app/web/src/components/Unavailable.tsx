import type { ReactNode } from 'react';

// Shown in place of a chart or try-it control when its data or checkpoints cannot be used.
export function Unavailable({ reason }: { reason: ReactNode }) {
  return (
    <div className="unavailable" role="status">
      <strong>Not available.</strong> <span>{reason}</span>
    </div>
  );
}

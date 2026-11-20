import type { ReactNode } from 'react';

// One-line caption placed under every chart.
export function WhatThisShows({ children }: { children: ReactNode }) {
  return (
    <p className="what-this-shows">
      <strong>What this shows:</strong> {children}
    </p>
  );
}

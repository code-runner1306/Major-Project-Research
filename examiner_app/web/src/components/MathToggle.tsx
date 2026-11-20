import type { ReactNode } from 'react';

// Collapsed "Show the maths" disclosure. Equations go inside it, never in the main explanation.
export function MathToggle({ children }: { children: ReactNode }) {
  return (
    <details className="math-toggle">
      <summary>Show the maths</summary>
      <div className="math-body">{children}</div>
    </details>
  );
}

import type { ReactNode } from 'react';
import './story.css';

/** The one bold line a section or step leads with (15 words at most). */
export function Takeaway({ children }: { children: ReactNode }) {
  return <p className="takeaway">{children}</p>;
}

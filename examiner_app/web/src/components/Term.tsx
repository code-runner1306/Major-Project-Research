import type { ReactNode } from 'react';
import { GLOSSARY } from '../glossary';

// Dotted-underline term. Hover or keyboard focus shows the plain-English definition.
export function Term({ term, children }: { term: string; children?: ReactNode }) {
  const entry = GLOSSARY[term];
  const label = children ?? entry?.term ?? term;
  if (!entry) return <span>{label}</span>;
  return (
    <span className="term" tabIndex={0} aria-label={`${entry.term}: ${entry.plain}`}>
      {label}
      <span className="term-tip" role="tooltip">
        <strong>{entry.term}</strong>
        <span>{entry.plain}</span>
      </span>
    </span>
  );
}

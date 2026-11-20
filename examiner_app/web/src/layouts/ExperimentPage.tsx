import type { ReactNode } from 'react';
import type { ExperimentDef, Section } from '../types';
import { LayoutChapters } from './LayoutChapters';
import { LayoutScroll } from './LayoutScroll';

export interface LayoutProps {
  def: ExperimentDef;
  sections: Section[];
}

// Experiment pages are slide decks (LayoutChapters). The start pages (Introduction, Datasets) keep the
// original scrolling chapters layout (LayoutScroll), per the user's 2026-11-20 instruction.
export function ExperimentPage({
  def,
  sections,
  notice,
}: {
  def: ExperimentDef;
  sections: Section[];
  notice?: ReactNode;
}) {
  if (def.group === 'start') {
    return (
      <>
        {notice}
        <LayoutScroll def={def} sections={sections} />
      </>
    );
  }
  return <LayoutChapters def={def} sections={sections} notice={notice} />;
}

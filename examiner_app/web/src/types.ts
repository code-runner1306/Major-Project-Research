// Shared contract between the shell, layouts and page content. See ../../SPEC.md before changing.
import type { ReactNode } from 'react';

export type SectionId = 'question' | 'model' | 'code' | 'results' | 'inside' | 'try';

export interface Section {
  id: SectionId;
  title: string;
  /** Plain-English explanation (layouts decide placement). */
  explain: ReactNode;
  /** The chart / diagram / interactive for this section. */
  visual: ReactNode;
}

export type ExperimentStatus = 'live' | 'results' | 'unavailable';
export type ExperimentGroup = 'start' | 'experiments' | 'appendix';

export interface ExperimentDef {
  id: string;
  title: string;
  group: ExperimentGroup;
  status: ExperimentStatus;
  /** "In one sentence" summary shown at the top of the page. */
  oneLiner: string;
  /** Everyday analogy for non-specialists. */
  analogy: string;
  /** Optional headline number for hero areas, e.g. { value: '0 / 6', label: 'A4 models with the shortcut' }. */
  headline?: { value: string; label: string };
}

export interface PageProps {
  def: ExperimentDef;
}

export interface CodeStep {
  title: string;
  plain: string;
  path: string;
  symbol?: string;
  lines?: { start: number; end: number };
  notes: { match: string; note: string }[];
}

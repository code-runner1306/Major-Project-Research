import type { PageProps, Section, SectionId } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { Unavailable } from '../components/Unavailable';

const PLACEHOLDER_SECTIONS: { id: SectionId; title: string }[] = [
  { id: 'question', title: 'Question' },
  { id: 'model', title: 'Model' },
  { id: 'code', title: 'Code' },
  { id: 'results', title: 'Results' },
  { id: 'inside', title: 'Inside' },
  { id: 'try', title: 'Try it' },
];

function placeholderSections(name: string): Section[] {
  return PLACEHOLDER_SECTIONS.map((s) => ({
    id: s.id,
    title: s.title,
    explain: (
      <p>
        Page in progress. This part of &ldquo;{name}&rdquo; will explain the {s.title.toLowerCase()} step in plain
        English once the real content is written.
      </p>
    ),
    visual: <div className="placeholder-visual">{s.title}</div>,
  }));
}

// Renders a not-yet-written experiment through the real layouts, so all three variants can be previewed.
export function PlaceholderPage({ def }: PageProps) {
  return (
    <ExperimentPage
      def={def}
      sections={placeholderSections(def.title)}
      notice={
        def.status === 'unavailable' ? (
          <Unavailable reason="This experiment is marked unavailable in this build, so no live results are shown." />
        ) : (
          <p className="page-progress">Page in progress.</p>
        )
      }
    />
  );
}

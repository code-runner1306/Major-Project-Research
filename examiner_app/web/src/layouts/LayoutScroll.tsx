import './scroll.css';
import { useReducedMotion } from 'motion/react';
import type { LayoutProps } from './ExperimentPage';
import { scrollToSection, useScrollSpy } from './scroll';
import { SectionErrorBoundary } from '../components/SectionErrorBoundary';

// Layout C: full-width hero, a sticky chapter bar, then one full-width band per section.
export function LayoutScroll({ def, sections }: LayoutProps) {
  const reduced = useReducedMotion();
  const active = useScrollSpy(sections.map((s) => s.id));

  return (
    <div className="chapters">
      <header className="hero">
        <h1>{def.title}</h1>
        <div className="hero-grid">
          <div>
            <div className="hero-label">In one sentence</div>
            <p className="hero-lede">{def.oneLiner}</p>
          </div>
          <div className="analogy-box">
            <h2>Think of it like this</h2>
            <p>{def.analogy}</p>
          </div>
          {def.headline && (
            <div className="hero-stat">
              <div className="stat-value">{def.headline.value}</div>
              <div className="stat-label">{def.headline.label}</div>
            </div>
          )}
        </div>
      </header>

      <nav className="chapter-bar" aria-label="Chapters">
        {sections.map((sec) => (
          <a
            key={sec.id}
            href={`#sec-${sec.id}`}
            className={sec.id === active ? 'chapter active' : 'chapter'}
            aria-current={sec.id === active ? 'location' : undefined}
            onClick={(e) => {
              e.preventDefault();
              scrollToSection(sec.id, reduced);
            }}
          >
            {sec.title}
          </a>
        ))}
      </nav>

      {sections.map((sec, i) => (
        <section
          key={sec.id}
          id={`sec-${sec.id}`}
          className={i % 2 === 1 ? 'band band-alt' : 'band'}
        >
          <div className="band-inner">
            <h2>{sec.title}</h2>
            <div className="explain">
              <SectionErrorBoundary>{sec.explain}</SectionErrorBoundary>
            </div>
            <div className="visual">
              <SectionErrorBoundary>{sec.visual}</SectionErrorBoundary>
            </div>
          </div>
        </section>
      ))}
    </div>
  );
}

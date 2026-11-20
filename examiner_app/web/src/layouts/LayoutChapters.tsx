import { useEffect, useRef, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import type { ExperimentDef, Section } from '../types';
import { SectionErrorBoundary } from '../components/SectionErrorBoundary';
import { setRouteSection, useHashRoute } from '../shell/useHashRoute';
import './deck.css';

interface LayoutChaptersProps {
  def: ExperimentDef;
  sections: Section[];
  /** Shown above the first slide (e.g. "Page in progress"). */
  notice?: ReactNode;
}

const SLIDE_OFFSET = 28;

// Slides slide in from the side they are moving towards and fade; `dir` is +1 forwards, -1 backwards.
const slideVariants = {
  enter: (dir: number) => ({ x: dir * SLIDE_OFFSET, opacity: 0 }),
  center: { x: 0, opacity: 1 },
  exit: (dir: number) => ({ x: dir * -SLIDE_OFFSET, opacity: 0 }),
};

/**
 * Slide deck: one section fills the stage at a time. The chapter bar, the bottom bar, PageDown/PageUp and ] / [
 * move between slides. The current slide lives in the hash (#/<experiment>/<section>).
 */
export function LayoutChapters({ def, sections, notice }: LayoutChaptersProps) {
  const route = useHashRoute();
  const reduce = useReducedMotion() ?? false;
  const count = sections.length;
  const found = route.id === def.id ? sections.findIndex((s) => s.id === route.section) : -1;
  const idx = found < 0 ? 0 : found;
  const current = sections[idx];
  // Direction of the last move (+1 forwards, -1 backwards), so the slide that leaves and the one that enters agree.
  const lastIdx = useRef(idx);
  const dirRef = useRef(1);
  if (lastIdx.current !== idx) {
    dirRef.current = idx > lastIdx.current ? 1 : -1;
    lastIdx.current = idx;
  }
  const dir = dirRef.current;

  const goTo = (target: number) => {
    const clamped = Math.min(count - 1, Math.max(0, target));
    if (clamped !== idx) setRouteSection(def.id, sections[clamped].id);
  };

  // Keyboard: number keys 1..n jump to that slide; PageDown / PageUp and ] / [ step. Typing in a text box,
  // select or editable element keeps its own keys (it never switches slides).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey) return;
      const t = e.target as HTMLElement | null;
      if (t && (t.isContentEditable || t.tagName === 'INPUT' || t.tagName === 'SELECT' || t.tagName === 'TEXTAREA')) return;
      if (/^[1-9]$/.test(e.key)) {
        const target = Number(e.key) - 1;
        if (target < count) {
          e.preventDefault();
          if (target !== idx) setRouteSection(def.id, sections[target].id);
        }
        return;
      }
      let step = 0;
      if (e.key === 'PageDown' || e.key === ']') step = 1;
      else if (e.key === 'PageUp' || e.key === '[') step = -1;
      else if (e.key === 'ArrowRight' || e.key === 'ArrowLeft') {
        // Arrow keys switch slides unless focus sits on a control that uses them itself
        // (in-slide tab rows, sliders, radio groups, listboxes).
        if (t?.closest('[role="tab"], [role="tablist"], [role="slider"], [role="radio"], [role="radiogroup"], [role="listbox"], [role="menu"]')) return;
        step = e.key === 'ArrowRight' ? 1 : -1;
      }
      if (step === 0) return;
      e.preventDefault();
      const target = Math.min(count - 1, Math.max(0, idx + step));
      if (target !== idx) setRouteSection(def.id, sections[target].id);
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [idx, count, def.id, sections]);

  const body = (
    <>
      {idx === 0 && (
        <header className="deck-hero">
          <div className="deck-hero__main">
            <h1>{def.title}</h1>
            <p className="deck-hero__lede">{def.oneLiner}</p>
          </div>
          <div className="deck-hero__analogy">
            <h2>Think of it like this</h2>
            <p>{def.analogy}</p>
          </div>
          {def.headline && (
            <div className="deck-hero__stat">
              <div className="stat-value">{def.headline.value}</div>
              <div className="stat-label">{def.headline.label}</div>
            </div>
          )}
          {notice && <div className="deck-notice">{notice}</div>}
        </header>
      )}
      <h2 className="deck-slide__title">{current.title}</h2>
      <div className="deck-slide__explain">
        <SectionErrorBoundary>{current.explain}</SectionErrorBoundary>
      </div>
      <div className="deck-slide__visual">
        <SectionErrorBoundary>{current.visual}</SectionErrorBoundary>
      </div>
    </>
  );

  return (
    <div className="deck">
      <nav className="deck-chapters" aria-label="Sections">
        {sections.map((s, i) => (
          <button
            key={s.id}
            type="button"
            className={i === idx ? 'deck-chapter is-active' : 'deck-chapter'}
            aria-current={i === idx ? 'step' : undefined}
            onClick={() => goTo(i)}
          >
            <span className="deck-chapter__num">{i + 1}</span>
            {s.title}
          </button>
        ))}
      </nav>

      <div className="deck-stage">
        <AnimatePresence mode="wait" initial={false} custom={dir}>
          {reduce ? (
            <section key={current.id} className="deck-slide" aria-label={current.title}>
              {body}
            </section>
          ) : (
            <motion.section
              key={current.id}
              className="deck-slide"
              aria-label={current.title}
              custom={dir}
              variants={slideVariants}
              initial="enter"
              animate="center"
              exit="exit"
              transition={{ duration: 0.22, ease: 'easeOut' }}
            >
              {body}
            </motion.section>
          )}
        </AnimatePresence>
      </div>

      <nav className="deck-bar" aria-label="Slide controls">
        <button type="button" className="deck-bar__btn" disabled={idx === 0} onClick={() => goTo(idx - 1)}>
          {idx > 0 ? `‹ Previous: ${sections[idx - 1].title}` : '‹ Previous'}
        </button>
        <span className="deck-bar__count" aria-live="polite">
          {idx + 1} / {count}
        </span>
        <button
          type="button"
          className="deck-bar__btn deck-bar__btn--next"
          disabled={idx === count - 1}
          onClick={() => goTo(idx + 1)}
        >
          {idx < count - 1 ? `Next: ${sections[idx + 1].title} ›` : 'Next ›'}
        </button>
      </nav>
    </div>
  );
}

import { useCallback, useEffect, useLayoutEffect, useRef, useState } from 'react';
import { motion, useReducedMotion, type Transition } from 'motion/react';
import { Remote, Segmented } from './common';
import { Takeaway } from '../../components/Takeaway';
import { WhatThisShows } from '../../components/WhatThisShows';
import { useElementSize } from '../../components/useElementSize';
import { useEvaluate } from './data';
import type { EvaluateRequest, EvaluateResponse, ModelRef, PickCard, PickSlot, ShortcutExample, ShortcutWrite } from './types';

// "Inside" story, three steps: one example, all examples, then move the question away.
// Every number and sentence is generated from POST /api/shortcut/evaluate (block-first layout, three rewrites).

const SHORTCUT_MODEL: ModelRef = { init: 'default', variant: 'scalar', seed: 44 };
const HEALTHY_MODEL: ModelRef = { init: 'default', variant: 'scalar', seed: 43 };

const COMPARE = [
  { id: 'shortcut', label: 'Shortcut model (seed 44)' },
  { id: 'healthy', label: 'Healthy model (seed 43)' },
] as const;
type CompareId = (typeof COMPARE)[number]['id'];
const MODEL_FOR: Record<CompareId, ModelRef> = { shortcut: SHORTCUT_MODEL, healthy: HEALTHY_MODEL };

const EXAMPLES = 64;
const SHOW = 6;
const EXAMPLE_SEED = 3;
const BASE_DISTANCE = 8;

const BLANKS = [
  { id: 0, label: '0' },
  { id: 4, label: '4' },
  { id: 8, label: '8' },
] as const;
type Blanks = (typeof BLANKS)[number]['id'];

const STEPS = ['One example', 'All 64 examples', 'Move the question further away'] as const;
type Step = 1 | 2 | 3;

const REWRITE_NAMES = ['the original value', 'the first rewrite', 'the second rewrite', 'the third rewrite', 'the fourth rewrite'];

// ------------------------------------------------------------------ sentences from data

/** Which write a memory card holds, in words: the original value, a rewrite, or another key's value. */
function describeCard(writes: ShortcutWrite[], slot: number): string {
  const card = writes.find((w) => w.slot === slot);
  if (!card) return 'a card';
  if (!card.is_target) return "a different key's value";
  const order = writes.filter((w) => w.is_target).findIndex((w) => w.slot === slot);
  return REWRITE_NAMES[order] ?? 'an earlier rewrite';
}

function oneExampleSentence(ex: ShortcutExample): string {
  const pick = ex.picked_slot;
  if (pick === null) return `The model answered ${ex.prediction}, a value not in memory. The newest value was ${ex.gold}.`;
  if (pick === ex.gold_slot) return `The model answered ${ex.prediction}, the newest value, from card ${pick}.`;
  return `The model answered ${ex.prediction}, card ${pick}, ${describeCard(ex.writes, pick)}, instead of ${ex.gold} on card ${ex.gold_slot}.`;
}

function topPick(picks: PickCard[]): PickCard | null {
  if (picks.length === 0) return null;
  return picks.reduce((best, p) => (p.count > best.count ? p : best));
}

function allExamplesSentence(top: PickCard, total: number): string {
  if (top.is_latest_target) return `In ${top.count} of ${total} examples it picked card ${top.slot}, the newest write.`;
  if (top.is_target) return `In ${top.count} of ${total} examples it picked card ${top.slot}. It is not reading the key — it grabs a position.`;
  return `In ${top.count} of ${total} examples it picked card ${top.slot}, another key's value. It counts positions.`;
}

function blanksSentence(top: PickCard, blanks: Blanks): string {
  const none = blanks === 0;
  if (top.is_latest_target) {
    return `${none ? 'With no blanks added' : `With ${blanks} blanks added`}, it still picks card ${top.slot}, the newest write.`;
  }
  const lead = none ? 'With no blanks added, it picks' : `With ${blanks} blanks added, it now picks`;
  if (top.is_target) return `${lead} card ${top.slot}, an older write of this key.`;
  return `${lead} card ${top.slot}, an unrelated key's value. It counts positions.`;
}

/** Non-null picks, in slot order, as cards. */
function cardsOf(rows: PickSlot[]): PickCard[] {
  return rows
    .filter((r): r is PickSlot & { slot: number } => r.slot !== null)
    .map((r) => ({ ...r, slot: r.slot, is_target: Boolean(r.is_target), is_latest_target: Boolean(r.is_latest_target) }))
    .sort((a, b) => a.slot - b.slot);
}

// ------------------------------------------------------------------ component

export function ErrorMap() {
  const [compare, setCompare] = useState<CompareId>('shortcut');
  const [step, setStep] = useState<Step>(1);
  const [blanks, setBlanks] = useState<Blanks>(0);
  const [index, setIndex] = useState(0);

  const req: EvaluateRequest = {
    ...MODEL_FOR[compare],
    layout: 'block_first',
    rewrites: 3,
    distance: BASE_DISTANCE + (step === 3 ? blanks : 0),
    examples: EXAMPLES,
    example_seed: EXAMPLE_SEED,
    patch: 'none',
    show_examples: SHOW,
  };
  const ev = useEvaluate(req);

  return (
    <div className="em">
      <div className="em-top">
        <nav className="em-steps" aria-label="Steps">
          {STEPS.map((title, i) => {
            const id = (i + 1) as Step;
            return (
              <button key={id} type="button" className="em-step" aria-pressed={step === id} onClick={() => setStep(id)}>
                <span className="em-step__num">{id}</span>
                <span className="em-step__title">{title}</span>
              </button>
            );
          })}
        </nav>
        <div className="em-compare">
          <Segmented options={[...COMPARE]} value={compare} onChange={setCompare} label="which model" />
        </div>
      </div>

      <Remote state={ev} what="the error map" tall>
        {(r) => {
          if (step === 1) return <OneExample examples={r.examples} index={index} onIndex={setIndex} />;
          if (step === 2) return <AllExamples r={r} />;
          return <MoveQuestion r={r} blanks={blanks} onBlanks={setBlanks} />;
        }}
      </Remote>
    </div>
  );
}

// ------------------------------------------------------------------ step 1

function OneExample({ examples, index, onIndex }: { examples: ShortcutExample[]; index: number; onIndex: (i: number) => void }) {
  const reduce = useReducedMotion() ?? false;
  if (examples.length === 0) return <p className="sc-muted">The server returned no examples for these settings.</p>;
  const n = examples.length;
  const at = Math.min(index, n - 1);
  const ex = examples[at];
  const sentenceDelay = reduce ? 0 : 2.9;

  return (
    <div className="em-step-body">
      <OneExampleDiagram key={at} ex={ex} />
      <motion.div
        className="em-sentence"
        initial={{ opacity: 0, y: 6 }}
        animate={{ opacity: 1, y: 0 }}
        transition={reduce ? { duration: 0 } : { delay: sentenceDelay, duration: 0.4, ease: 'easeOut' }}
      >
        <Takeaway>{oneExampleSentence(ex)}</Takeaway>
      </motion.div>
      <div className="em-nav">
        <button type="button" className="sc-btn" onClick={() => onIndex((at - 1 + n) % n)}>
          ‹ Previous example
        </button>
        <button type="button" className="sc-btn" onClick={() => onIndex((at + 1) % n)}>
          Next example ›
        </button>
        <span className="sc-muted">
          Example {at + 1} of {n}
        </span>
      </div>
      <WhatThisShows>cards are memory in write order. The arrow shows which card the model&apos;s answer came from.</WhatThisShows>
    </div>
  );
}

interface ArrowGeometry {
  d: string;
  labelX: number;
  labelY: number;
}

/**
 * The memory cards as HTML, the question under them, and one arrow. The arrow is an SVG overlay drawn from the
 * measured card and question boxes, so it meets their real edges at any size.
 */
function OneExampleDiagram({ ex }: { ex: ShortcutExample }) {
  const reduce = useReducedMotion() ?? false;
  const [measureRef, box] = useElementSize<HTMLDivElement>();
  const rootEl = useRef<HTMLDivElement | null>(null);
  const cardEls = useRef<(HTMLDivElement | null)[]>([]);
  const questionEl = useRef<HTMLDivElement | null>(null);
  const [arrow, setArrow] = useState<ArrowGeometry | null>(null);

  const setRoot = useCallback(
    (node: HTMLDivElement | null) => {
      rootEl.current = node;
      measureRef(node);
    },
    [measureRef],
  );

  const writes = ex.writes;
  const pick = ex.picked_slot;
  const right = pick !== null && pick === ex.gold_slot;
  const tone = right ? 'is-ok' : 'is-bad';

  const at = (delay: number, duration = 0.4): Transition => (reduce ? { duration: 0 } : { delay, duration, ease: 'easeOut' });
  const cardsDone = reduce ? 0 : 0.1 * writes.length + 0.35;
  const badgeAt = cardsDone + 0.15;
  const arrowAt = badgeAt + 0.5;
  const labelAt = arrowAt + 0.6;

  // Re-measure after every render, and whenever the root, the question or a card changes size (CSS can move them
  // without the root changing size).
  const measure = useCallback(() => {
    const root = rootEl.current;
    const question = questionEl.current;
    const pickIndex = writes.findIndex((w) => w.slot === pick);
    const card = pickIndex >= 0 ? cardEls.current[pickIndex] : null;
    if (!root || !question || !card) {
      setArrow(null);
      return;
    }
    const base = root.getBoundingClientRect();
    const q = question.getBoundingClientRect();
    const c = card.getBoundingClientRect();
    const sx = q.left + q.width / 2 - base.left;
    const sy = q.top - base.top - 4;
    const tx = c.left + c.width / 2 - base.left;
    const ty = c.bottom - base.top + 6;
    const mid = (sy + ty) / 2;
    const next: ArrowGeometry = { d: `M ${sx} ${sy} C ${sx} ${mid}, ${tx} ${mid}, ${tx} ${ty}`, labelX: Math.max(sx, tx) + 12, labelY: mid + 4 };
    setArrow((prev) => (prev && prev.d === next.d && prev.labelX === next.labelX && prev.labelY === next.labelY ? prev : next));
  }, [writes, pick]);

  useLayoutEffect(() => {
    measure();
  });

  useEffect(() => {
    const root = rootEl.current;
    if (!root) return;
    const observer = new ResizeObserver(() => measure());
    observer.observe(root);
    if (questionEl.current) observer.observe(questionEl.current);
    cardEls.current.forEach((el) => el && observer.observe(el));
    return () => observer.disconnect();
  }, [measure, writes]);

  return (
    <div
      className="em-diagram"
      ref={setRoot}
      role="img"
      aria-label={`Memory cards in write order. The model answered ${ex.prediction}, the newest value is ${ex.gold}.`}
    >
      <div className="em-cards">
        {writes.map((w, i) => (
          <motion.div
            key={w.slot}
            className="em-cardcol"
            ref={(el: HTMLDivElement | null) => {
              cardEls.current[i] = el;
            }}
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            transition={at(reduce ? 0 : 0.1 * i)}
          >
            {ex.gold_slot === w.slot && (
              <div className="em-badge-wrap">
                <motion.div className="em-badge" initial={{ opacity: 0 }} animate={{ opacity: 1 }} transition={at(badgeAt, 0.3)}>
                  <svg viewBox="0 0 12 12" aria-hidden="true">
                    <path d="M2 6.5 L5 9.5 L10 2.5" />
                  </svg>
                  correct answer
                </motion.div>
              </div>
            )}
            <div className={w.is_target ? 'em-mem is-same' : 'em-mem'}>
              <span className="em-mem__card">card {w.slot}</span>
              <span className="em-mem__key">key {w.key}</span>
              <span className="em-mem__value">{w.value}</span>
              <span className="em-mem__tags">
                {w.is_target && <span className="em-tag">same key</span>}
                {w.is_latest_target && <span className="em-tag is-solid">newest</span>}
              </span>
            </div>
          </motion.div>
        ))}
      </div>

      <motion.div
        ref={questionEl}
        className="em-question"
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={at(reduce ? 0 : 0.1 * writes.length)}
      >
        What is the newest value for key {ex.question_key}?
      </motion.div>

      <svg className="em-arrow-svg" width={box.width} height={box.height} viewBox={`0 0 ${box.width} ${box.height}`} aria-hidden="true">
        <defs>
          <marker id="em-head-ok" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="em-head is-ok" />
          </marker>
          <marker id="em-head-bad" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="5" markerHeight="5" orient="auto">
            <path d="M0,0 L10,5 L0,10 z" className="em-head is-bad" />
          </marker>
        </defs>
        {arrow && (
          <>
            <motion.path
              d={arrow.d}
              fill="none"
              strokeWidth={4}
              className={`em-arrow ${tone}`}
              markerEnd={right ? 'url(#em-head-ok)' : 'url(#em-head-bad)'}
              initial={{ pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={at(arrowAt, 0.7)}
            />
            <motion.text
              x={arrow.labelX}
              y={arrow.labelY}
              className={`em-arrow-label ${tone}`}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              transition={at(labelAt, 0.3)}
            >
              model&apos;s answer
            </motion.text>
          </>
        )}
      </svg>
    </div>
  );
}

// ------------------------------------------------------------------ steps 2 and 3

function AllExamples({ r }: { r: EvaluateResponse }) {
  const picks = cardsOf(r.pick_histogram);
  const top = topPick(picks);
  const nothing = r.pick_histogram.find((row) => row.slot === null)?.count ?? 0;
  return (
    <div className="em-step-body">
      <PickBars picks={picks} total={r.n_examples} />
      {nothing > 0 && <p className="sc-muted">{nothing} answers matched no card.</p>}
      {top && (
        <div className="em-sentence">
          <Takeaway>{allExamplesSentence(top, r.n_examples)}</Takeaway>
        </div>
      )}
      <WhatThisShows>each bar counts how many of the 64 answers came from that card.</WhatThisShows>
    </div>
  );
}

function MoveQuestion({ r, blanks, onBlanks }: { r: EvaluateResponse; blanks: Blanks; onBlanks: (b: Blanks) => void }) {
  const picks = cardsOf(r.pick_histogram);
  const top = topPick(picks);
  return (
    <div className="em-step-body">
      <div className="em-blanks">
        <span className="sc-field__label">0 / 4 / 8 extra blank tokens before the question</span>
        <Segmented options={[...BLANKS]} value={blanks} onChange={onBlanks} label="extra blank tokens before the question" />
      </div>
      <PickBars picks={picks} total={r.n_examples} />
      {top && (
        <div className="em-sentence">
          <Takeaway>{blanksSentence(top, blanks)}</Takeaway>
        </div>
      )}
      <WhatThisShows>the same bars, after the question moves further from memory. Only the blank count changes.</WhatThisShows>
    </div>
  );
}

/** Columns of cards with a bar above each: the bar height is the share of examples that came from that card. */
function PickBars({ picks, total }: { picks: PickCard[]; total: number }) {
  const reduce = useReducedMotion() ?? false;
  const transition: Transition = reduce ? { duration: 0 } : { duration: 0.8, ease: 'easeOut' };
  if (picks.length === 0) return <p className="sc-muted">No answers to draw.</p>;
  return (
    <div className="em-bars" style={{ gridTemplateColumns: `repeat(${picks.length}, minmax(0, 1fr))` }}>
      {picks.map((p) => {
        const share = total > 0 ? p.count / total : 0;
        return (
          <div key={p.slot} className="em-col">
            <div className="em-col__count">{p.count}</div>
            <div className="em-col__track">
              <motion.div
                className={p.is_latest_target ? 'em-col__bar is-correct' : 'em-col__bar is-wrong'}
                initial={{ height: 0 }}
                animate={{ height: `${share * 100}%` }}
                transition={transition}
              />
            </div>
            <div className={p.is_target ? 'em-card is-same' : 'em-card'}>
              <span className="em-card__name">card {p.slot}</span>
              {p.is_target && <span className="em-card__tag">same key</span>}
              {p.is_latest_target && <span className="em-card__tag is-solid">correct</span>}
            </div>
          </div>
        );
      })}
    </div>
  );
}

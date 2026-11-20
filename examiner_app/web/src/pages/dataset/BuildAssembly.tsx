import { useEffect, useMemo, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { Term } from '../../components/Term';
import { WhatThisShows } from '../../components/WhatThisShows';
import {
  ExampleError,
  RoleLegend,
  SeedControl,
  Skeleton,
  WaitingForServer,
  isKeyRole,
  isValueRole,
  numberWord,
  useLiveExample,
} from './shared';
import type { DatasetExample } from './types';

// Builds one live overwrite example step by step.

interface Write {
  keyPos: number;
  valPos: number;
  target: boolean;
}

interface Plan {
  writes: Write[];
  /** One write per distinct key, in the order the keys first appear. */
  firstWrites: Write[];
  /** The second write of the key that is written twice (the rewrite), if any. */
  rewrite: Write | null;
  keyTokens: number[];
  writePositions: number[];
  fillerPositions: number[];
  restPositions: number[];
  scored: number[];
}

/** Splits an overwrite example into its writes (key token followed by value token) and the other positions. */
export function buildPlan(ex: DatasetExample): Plan {
  const writes: Write[] = [];
  for (let i = 0; i + 1 < ex.roles.length; i++) {
    if (isKeyRole(ex.roles[i]) && isValueRole(ex.roles[i + 1])) {
      writes.push({ keyPos: i, valPos: i + 1, target: ex.roles[i] === 'target_key' });
      i++;
    }
  }
  const byKey = new Map<number, Write[]>();
  for (const w of writes) {
    const k = ex.tokens[w.keyPos];
    byKey.set(k, [...(byKey.get(k) ?? []), w]);
  }
  const groups = [...byKey.values()];
  const repeated = groups.find((ws) => ws.length > 1);
  const writePositions = writes.flatMap((w) => [w.keyPos, w.valPos]).sort((a, b) => a - b);
  const fillerPositions = ex.roles.flatMap((r, i) => (r === 'filler' ? [i] : []));
  const used = new Set([...writePositions, ...fillerPositions]);
  const restPositions = ex.tokens.map((_, i) => i).filter((i) => !used.has(i));
  return {
    writes,
    firstWrites: groups.map((ws) => ws[0]),
    rewrite: repeated ? repeated[1] : null,
    keyTokens: [...byKey.keys()],
    writePositions,
    fillerPositions,
    restPositions,
    scored: ex.scored.map((s) => s.position),
  };
}

const STEP_TITLES = [
  'Draw the keys',
  'Give each key a value',
  'Rewrite one key',
  'Order the writes',
  'Pad with filler',
  'Append the queries',
  'Mark the scored answers',
];
const LAST = STEP_TITLES.length - 1;

/** Tokens sitting in the tray (drawn but not yet placed) at a given step. */
function trayAt(plan: Plan, stage: number): number[] {
  if (stage >= 3) return [];
  const out: number[] = [];
  for (const w of plan.firstWrites) {
    out.push(w.keyPos);
    if (stage >= 1) out.push(w.valPos);
  }
  if (stage >= 2 && plan.rewrite) out.push(plan.rewrite.keyPos, plan.rewrite.valPos);
  return out;
}

/** Positions already placed in the strip at a given step. */
function placedAt(plan: Plan, stage: number): Set<number> {
  const placed = new Set<number>();
  if (stage >= 3) plan.writePositions.forEach((p) => placed.add(p));
  if (stage >= 4) plan.fillerPositions.forEach((p) => placed.add(p));
  if (stage >= 5) plan.restPositions.forEach((p) => placed.add(p));
  return placed;
}

function sentenceFor(stage: number, plan: Plan, ex: DatasetExample): string {
  const keys = plan.keyTokens;
  const rw = plan.rewrite;
  switch (stage) {
    case 0:
      return `Draw ${numberWord(keys.length)} different keys (${keys.join(', ')}), the labels the model will later look up.`;
    case 1:
      return 'Give each key its first value, so every key now holds one stored value.';
    case 2:
      return rw
        ? `Write key ${ex.tokens[rw.keyPos]} again with the new value ${ex.tokens[rw.valPos]}, so it now has two values and only the newer one counts.`
        : 'This example has no rewrite, so every key is written exactly once.';
    case 3: {
      if (!rw) return `Put the ${numberWord(plan.writes.length)} writes in order.`;
      const target = ex.tokens[rw.keyPos];
      const after = plan.writes.filter((w) => w.keyPos > rw.keyPos && ex.tokens[w.keyPos] !== target).length;
      return `Put the ${numberWord(plan.writes.length)} writes in order, so ${numberWord(after)} unrelated write${after === 1 ? ' comes' : 's come'} after the latest write to key ${target}.`;
    }
    case 4:
      return plan.fillerPositions.length > 0
        ? `Fill the empty positions with ${numberWord(plan.fillerPositions.length)} filler tokens, which carry no information.`
        : 'No filler is needed: the writes already fill the example.';
    case 5:
      return `Append ${numberWord(plan.restPositions.length)} query tokens: a marker, then the keys the model is asked about.`;
    default:
      return `Mark the ${numberWord(plan.scored.length)} scored answers: the model is graded only at these positions.`;
  }
}

function Chip({
  pos,
  ex,
  scored,
  reduce,
}: {
  pos: number;
  ex: DatasetExample;
  scored: boolean;
  reduce: boolean;
}) {
  const role = ex.roles[pos];
  const token = ex.tokens[pos];
  const text = role === 'filler' && !scored ? '·' : role === 'marker' ? '?' : String(token);
  const classes = ['viz-token', `viz-token--${role}`, scored ? 'viz-token--scored' : ''].filter(Boolean).join(' ');
  return (
    <motion.span
      layoutId={`ds-asm-tok-${pos}`}
      className={classes}
      title={`position ${pos}`}
      initial={reduce ? false : { opacity: 0, scale: 0.6 }}
      animate={{ opacity: 1, scale: 1 }}
      transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 240, damping: 26 }}
    >
      {text}
    </motion.span>
  );
}

function AssemblyStage({ ex }: { ex: DatasetExample }) {
  const plan = useMemo(() => buildPlan(ex), [ex]);
  const reduce = useReducedMotion() ?? false;
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [shownEx, setShownEx] = useState(ex);

  // A new example starts again from the first step (state is adjusted during render, not in an effect).
  if (shownEx !== ex) {
    setShownEx(ex);
    setStage(0);
    setPlaying(false);
  }

  // Play runs only while there are steps left: it stops by itself at the last step.
  const isPlaying = playing && stage < LAST;
  useEffect(() => {
    if (!isPlaying) return;
    const timer = window.setTimeout(() => setStage((s) => s + 1), reduce ? 700 : 1500);
    return () => window.clearTimeout(timer);
  }, [isPlaying, stage, reduce]);

  const tray = trayAt(plan, stage);
  const placed = placedAt(plan, stage);
  const strip = ex.tokens.map((_, i) => i).filter((i) => placed.has(i));
  const scoredNow = stage >= LAST ? new Set(plan.scored) : new Set<number>();

  return (
    <div className="ds-asm">
      <ol className="ds-asm__steps" aria-label="construction steps">
        {STEP_TITLES.map((title, i) => (
          <li key={title}>
            <button
              type="button"
              className={i === stage ? 'ds-asm__step is-on' : i < stage ? 'ds-asm__step is-done' : 'ds-asm__step'}
              aria-current={i === stage ? 'step' : undefined}
              onClick={() => {
                setPlaying(false);
                setStage(i);
              }}
            >
              <span className="ds-asm__num">{i + 1}</span>
              {title}
            </button>
          </li>
        ))}
      </ol>

      <p className="ds-asm__sentence" aria-live="polite">
        <strong>
          Step {stage + 1} of {STEP_TITLES.length}: {STEP_TITLES[stage]}.
        </strong>{' '}
        {sentenceFor(stage, plan, ex)}
      </p>

      <div className="ds-asm__lane">
        <span className="ds-asm__lane-label">Drawn, not yet placed</span>
        <div className="ds-asm__tray" aria-label="tray">
          {tray.length === 0 && <span className="ds-muted">Nothing waiting.</span>}
          {tray.map((pos) => (
            <Chip key={pos} pos={pos} ex={ex} scored={false} reduce={reduce} />
          ))}
        </div>
      </div>

      <div className="ds-asm__lane">
        <span className="ds-asm__lane-label">The example so far</span>
        <div className="ds-asm__strip" aria-label="example so far">
          {strip.map((pos) => (
            <Chip key={pos} pos={pos} ex={ex} scored={scoredNow.has(pos)} reduce={reduce} />
          ))}
        </div>
      </div>

      <div className="ds-asm__controls">
        <button
          type="button"
          className="ds-btn"
          aria-pressed={isPlaying}
          disabled={stage >= LAST}
          onClick={() => setPlaying((p) => !p)}
        >
          {isPlaying ? 'Pause' : 'Play'}
        </button>
        <button
          type="button"
          className="ds-btn"
          disabled={stage >= LAST}
          onClick={() => {
            setPlaying(false);
            setStage((s) => Math.min(LAST, s + 1));
          }}
        >
          Step
        </button>
        <button
          type="button"
          className="ds-btn"
          onClick={() => {
            setPlaying(false);
            setStage(0);
          }}
        >
          Restart
        </button>
      </div>

      <RoleLegend roles={ex.roles} />
    </div>
  );
}

export function BuildAssembly() {
  const [seed, setSeed] = useState(7);
  const ex = useLiveExample('overwrite', seed);

  return (
    <div className="ds-asm-wrap">
      <p className="ds-muted">
        Built from a live <Term term="overwrite-task" /> example. Each step draws a piece, and the pieces fly into the example in their real
        positions.
      </p>
      <SeedControl seed={seed} onChange={setSeed} />
      {ex.error && <ExampleError error={ex.error} />}
      {ex.waiting && <WaitingForServer />}
      {!ex.data && !ex.error && !ex.waiting && <Skeleton label="loading the overwrite example" tall />}
      {ex.data && <AssemblyStage ex={ex.data} />}
      <WhatThisShows>
        the top row holds the pieces not yet placed; the strip below is the example so far. Each move is a real token from the server, placed where
        the generator put it. Outlined tokens are scored answers.
      </WhatThisShows>
    </div>
  );
}

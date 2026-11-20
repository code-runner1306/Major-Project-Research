import { useEffect, useMemo, useRef, useState } from 'react';
import { animate, motion, useReducedMotion } from 'motion/react';
import type { Variant } from '../../api';
import { MathToggle } from '../../components/MathToggle';
import { WhatThisShows } from '../../components/WhatThisShows';
import { parseRgb, useThemeTokens } from '../../theme';
import { makeKeyValue, makeMemory, runStep, type Mat, type Vec } from './toyMath';

const ACCENT = 'var(--accent)';
const CHANNEL = 'var(--c-channel)';
// Cell colours are animated by motion, which needs resolved values rather than var().
const CELL_TOKENS = ['--surface', '--accent', '--text-muted', '--text', '--on-accent'] as const;
const MEMORY = makeMemory(7);
const STAGE_LABELS = ['Start', 'Fade', 'Read', 'Write'];
const STAGE_TEXT = [
  'The memory as it stands before this token arrives.',
  'Step 1, fade: each column is multiplied by its own decay gate α. In scalar mode every column uses the same α.',
  'Step 2, read and compare: the faded memory returns S̃k for this key. The gap to the wanted value is r = v − S̃k.',
  'Step 3, write: add β · r kᵀ. With β = 1 and a unit-length key, the memory now returns v for this key exactly.',
];

const mix = (a: number[], b: number[], t: number) => a.map((x, i) => x + (b[i] - x) * t);

/** Accent for positive entries, muted text colour for negative, surface colour at zero. */
function cellColour(v: number, tk: Record<(typeof CELL_TOKENS)[number], string>): string {
  const t = Math.min(1, Math.abs(v));
  const [r, g, b] = mix(parseRgb(tk['--surface']), parseRgb(v >= 0 ? tk['--accent'] : tk['--text-muted']), t);
  return `rgb(${Math.round(r)},${Math.round(g)},${Math.round(b)})`;
}

const fmt = (x: number) => (Math.abs(x) < 0.005 ? '0.00' : x.toFixed(2));

/** Number that tweens to its new value. */
function TweenNumber({ value, reduce }: { value: number; reduce: boolean }) {
  const [shown, setShown] = useState(value);
  const fromRef = useRef(value);
  useEffect(() => {
    if (reduce) {
      fromRef.current = value;
      setShown(value);
      return;
    }
    const controls = animate(fromRef.current, value, {
      duration: 0.6,
      ease: 'easeOut',
      onUpdate: (v) => {
        fromRef.current = v;
        setShown(v);
      },
    });
    return () => controls.stop();
  }, [value, reduce]);
  return <>{Math.abs(shown) < 0.00005 ? '0.0000' : shown.toFixed(4)}</>;
}

function MemoryGrid({ matrix, reduce, alphaRow, alphaColour }: { matrix: Mat; reduce: boolean; alphaRow?: number[]; alphaColour?: string }) {
  const tk = useThemeTokens(CELL_TOKENS);
  return (
    <div className="intro-toy__grid-wrap">
      {alphaRow && (
        <div className="intro-toy__alpha-row" aria-label="Decay gate α per column">
          {alphaRow.map((a, j) => (
            <motion.span
              key={j}
              className="intro-toy__alpha"
              style={{ borderColor: alphaColour, color: alphaColour }}
              animate={{ opacity: 1 }}
              initial={{ opacity: 0 }}
              transition={{ duration: reduce ? 0 : 0.4 }}
            >
              α {a.toFixed(2)}
            </motion.span>
          ))}
        </div>
      )}
      <div className="intro-toy__grid" role="img" aria-label="4 by 4 memory matrix">
        {matrix.flatMap((row, i) =>
          row.map((v, j) => (
            <motion.div
              key={`${i}-${j}`}
              className="intro-toy__cell"
              animate={{ backgroundColor: cellColour(v, tk), color: Math.abs(v) > 0.6 ? tk['--on-accent'] : tk['--text'] }}
              transition={{ duration: reduce ? 0 : 0.6, ease: 'easeOut' }}
            >
              {fmt(v)}
            </motion.div>
          )),
        )}
      </div>
    </div>
  );
}

/** Vertical bars around a centre line; positive values rise, negative values fall. */
function VBars({ values, label, tone, reduce }: { values: Vec; label: string; tone: 'ink' | 'soft'; reduce: boolean }) {
  return (
    <div className="intro-vbars" role="img" aria-label={label}>
      {values.map((v, i) => {
        const h = Math.min(1, Math.abs(v)) * 44;
        const up = v >= 0;
        return (
          <div key={i} className="intro-vbars__col">
            <div className="intro-vbars__track">
              <motion.div
                className={`intro-vbars__bar intro-vbars__bar--${tone}`}
                style={up ? { bottom: '50%' } : { top: '50%' }}
                animate={{ height: h }}
                transition={{ duration: reduce ? 0 : 0.6, ease: 'easeOut' }}
              />
            </div>
            <span className="intro-vbars__num">{fmt(v)}</span>
          </div>
        );
      })}
    </div>
  );
}

function Slider({
  label,
  value,
  min,
  max,
  step,
  colour,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  colour?: string;
  onChange: (v: number) => void;
}) {
  return (
    <label className="intro-slider">
      <span>
        {label} <strong>{value.toFixed(2)}</strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        style={colour ? { accentColor: colour } : undefined}
        onChange={(e) => onChange(Number(e.target.value))}
      />
    </label>
  );
}

/**
 * Four-by-four delta-rule toy, computed in the browser. Shows the three steps (fade, read, write) in
 * sequence with Play, or one at a time with Next step. `compact` hides the sliders.
 */
export function ToyDeltaRule({ compact = false }: { compact?: boolean }) {
  const reduce = useReducedMotion() ?? false;
  const [variant, setVariant] = useState<Variant>('scalar');
  const [scalarAlpha, setScalarAlpha] = useState(0.85);
  const [channelAlpha, setChannelAlpha] = useState<number[]>([0.95, 0.8, 0.55, 0.25]);
  const [beta, setBeta] = useState(1);
  const [seed, setSeed] = useState(11);
  const [stage, setStage] = useState(0);
  const [playing, setPlaying] = useState(false);

  const { key, value } = useMemo(() => makeKeyValue(seed), [seed]);
  const alpha = variant === 'scalar' ? [scalarAlpha, scalarAlpha, scalarAlpha, scalarAlpha] : channelAlpha;
  const run = runStep(MEMORY, alpha, key, value, beta);
  const colour = variant === 'scalar' ? ACCENT : CHANNEL;

  // Play: advance one stage every 1.4 s (faster with reduced motion) until the write is done.
  useEffect(() => {
    if (!playing) return;
    if (stage >= 3) {
      setPlaying(false);
      return;
    }
    const t = window.setTimeout(() => setStage((s) => s + 1), reduce ? 900 : 1400);
    return () => window.clearTimeout(t);
  }, [playing, stage, reduce]);

  const play = () => {
    setStage(0);
    setPlaying(true);
  };
  const next = () => {
    setPlaying(false);
    setStage((s) => Math.min(3, s + 1));
  };
  const reset = () => {
    setPlaying(false);
    setStage(0);
  };

  const shownMemory = stage === 0 ? MEMORY : stage === 3 ? run.next : run.faded;
  const readVec = stage >= 3 ? run.readAfter : run.read;
  const readLabel = stage >= 3 ? 'Memory returns after the write, S′k' : 'Memory returns, S̃k';

  return (
    <div className="intro-toy">
      <div className="intro-toy__controls">
        <div className="intro-seg" role="radiogroup" aria-label="Decay granularity">
          {(['scalar', 'channel-wise'] as Variant[]).map((v) => (
            <button
              key={v}
              type="button"
              role="radio"
              aria-checked={variant === v}
              className={`intro-seg__btn${variant === v ? ' is-on' : ''}`}
              style={variant === v ? { background: v === 'scalar' ? ACCENT : CHANNEL, borderColor: v === 'scalar' ? ACCENT : CHANNEL } : undefined}
              onClick={() => setVariant(v)}
            >
              {v === 'scalar' ? 'Scalar decay' : 'Channel-wise decay'}
            </button>
          ))}
        </div>
        <div className="intro-toy__buttons">
          <button type="button" className="intro-btn intro-btn--primary" onClick={play} disabled={playing}>
            Play
          </button>
          <button type="button" className="intro-btn" onClick={next} disabled={stage >= 3 || playing}>
            Next step
          </button>
          <button type="button" className="intro-btn" onClick={reset}>
            Reset
          </button>
          <button
            type="button"
            className="intro-btn"
            onClick={() => {
              reset();
              setSeed((s) => s + 1);
            }}
          >
            New key and value
          </button>
        </div>
      </div>

      {!compact && (
        <div className="intro-toy__sliders">
          {variant === 'scalar' ? (
            <Slider
              label="Decay gate α (one number for every column)"
              value={scalarAlpha}
              min={0}
              max={1}
              step={0.01}
              colour={colour}
              onChange={setScalarAlpha}
            />
          ) : (
            channelAlpha.map((a, j) => (
              <Slider
                key={j}
                label={`Decay gate α, column ${j + 1}`}
                value={a}
                min={0}
                max={1}
                step={0.01}
                colour={colour}
                onChange={(v) => setChannelAlpha((prev) => prev.map((x, i) => (i === j ? v : x)))}
              />
            ))
          )}
          <Slider label="Write gate β" value={beta} min={0} max={1} step={0.01} onChange={setBeta} />
        </div>
      )}

      <ol className="intro-toy__stages" aria-label="Steps">
        {STAGE_LABELS.map((label, i) => (
          <li key={label} className={i === stage ? 'is-current' : i < stage ? 'is-done' : undefined}>
            {label}
          </li>
        ))}
      </ol>
      <p className="intro-toy__caption" aria-live="polite">
        {STAGE_TEXT[stage]}
      </p>

      <div className="intro-toy__stage">
        <section className="intro-toy__panel">
          <h4>Memory S (4 × 4)</h4>
          <MemoryGrid
            matrix={shownMemory}
            reduce={reduce}
            alphaRow={stage >= 1 ? alpha : undefined}
            alphaColour={colour}
          />
        </section>

        <section className="intro-toy__panel">
          <h4>Key k (unit length)</h4>
          <VBars values={key} label="Key vector" tone="ink" reduce={reduce} />
        </section>

        <motion.section
          className="intro-toy__panel"
          animate={{ opacity: stage >= 2 ? 1 : 0.35 }}
          transition={{ duration: reduce ? 0 : 0.4 }}
        >
          <h4>Compare with the wanted value v</h4>
          <div className="intro-toy__compare">
            <div>
              <span className="intro-toy__legend">Wanted v</span>
              <VBars values={value} label="Wanted value" tone="ink" reduce={reduce} />
            </div>
            <div>
              <span className="intro-toy__legend">{readLabel}</span>
              <VBars values={readVec} label="Memory read" tone="soft" reduce={reduce} />
            </div>
            <div>
              <span className="intro-toy__legend">Difference r = v − S̃k</span>
              <VBars values={run.residual} label="Difference" tone="soft" reduce={reduce} />
            </div>
          </div>
        </motion.section>
      </div>

      <div className="intro-toy__errors">
        <div>
          <span>Read error before the write</span>
          <strong>
            <TweenNumber value={run.errBefore} reduce={reduce} />
          </strong>
          <small>‖v − S̃k‖</small>
        </div>
        <div>
          <span>Read error after the write</span>
          <strong>{stage >= 3 ? <TweenNumber value={run.errAfter} reduce={reduce} /> : '—'}</strong>
          <small>‖v − S′k‖</small>
        </div>
      </div>

      <WhatThisShows>
        the memory grid after each step, the key, the value the memory should return and its read error. With β = 1 the
        error after the write is zero; change β to see it stop short.
      </WhatThisShows>

      <MathToggle>
        <pre className="intro-math">{`Fade:   S̃ = S · diag(α)        (column j is multiplied by α_j)
Read:   S̃k,  r = v − S̃k
Write:  S′ = S̃ + β · r kᵀ
Check:  S′k = S̃k + β · r (kᵀk) = S̃k + β · r      since kᵀk = 1
        so S′k = v when β = 1`}</pre>
      </MathToggle>
    </div>
  );
}

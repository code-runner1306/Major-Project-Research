import { useEffect, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { WhatThisShows } from '../../components/WhatThisShows';

const N = 8;
const SCALAR = 'var(--c-scalar)';
const CHANNEL = 'var(--c-channel)';
const SCALAR_PERIOD = 7;
type Mode = 'scalar' | 'channel-wise';

// Fixed "ink" strength per cell, so the board looks written-on rather than uniform.
const INK = Array.from({ length: N * N }, (_, i) => {
  const x = Math.sin(i * 12.9898) * 43758.5453;
  return 0.6 + 0.4 * (x - Math.floor(x));
});
// Each column gets its own period and phase in channel-wise mode, so columns fade at different rates.
const PERIOD = Array.from({ length: N }, (_, c) => 4.5 + c * 0.9);
const PHASE = Array.from({ length: N }, (_, c) => c * 0.8);

/** Dimmer level in [0, 1] that cycles smoothly. */
function level(t: number, period: number, phase: number): number {
  return 0.5 + 0.5 * Math.sin((2 * Math.PI * t) / period + phase);
}

/**
 * Whiteboard analogy. An 8 x 8 board of ink cells. Scalar decay is one dimmer that fades the whole board evenly;
 * channel-wise decay is one dimmer per column, so columns fade at different rates. Loops gently; static with reduced motion.
 */
export function WhiteboardAnimation() {
  const reduce = useReducedMotion() ?? false;
  const [mode, setMode] = useState<Mode>('scalar');
  const cells = useRef<(HTMLDivElement | null)[]>([]);
  const dimmers = useRef<(HTMLDivElement | null)[]>([]);

  useEffect(() => {
    const paint = (t: number) => {
      for (let c = 0; c < N; c++) {
        const lv = mode === 'scalar' ? level(t, SCALAR_PERIOD, 0) : level(t, PERIOD[c], PHASE[c]);
        for (let r = 0; r < N; r++) {
          const cell = cells.current[r * N + c];
          if (cell) cell.style.opacity = String(0.08 + 0.92 * lv * INK[r * N + c]);
        }
        if (mode === 'channel-wise') {
          const dim = dimmers.current[c];
          if (dim) dim.style.height = `${Math.round(lv * 100)}%`;
        }
      }
      if (mode === 'scalar') {
        const dim = dimmers.current[0];
        if (dim) dim.style.height = `${Math.round(level(t, SCALAR_PERIOD, 0) * 100)}%`;
      }
    };

    if (reduce) {
      paint(2);
      return;
    }
    let raf = 0;
    const start = performance.now();
    const tick = (now: number) => {
      paint((now - start) / 1000);
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [mode, reduce]);

  const colour = mode === 'scalar' ? SCALAR : CHANNEL;
  const dimmerCount = mode === 'scalar' ? 1 : N;

  return (
    <div className="intro-board">
      <div className="intro-seg" role="radiogroup" aria-label="Decay granularity">
        {(['scalar', 'channel-wise'] as Mode[]).map((m) => (
          <button
            key={m}
            type="button"
            role="radio"
            aria-checked={mode === m}
            className={`intro-seg__btn${mode === m ? ' is-on' : ''}`}
            style={mode === m ? { background: m === 'scalar' ? SCALAR : CHANNEL, borderColor: m === 'scalar' ? SCALAR : CHANNEL } : undefined}
            onClick={() => setMode(m)}
          >
            {m === 'scalar' ? 'Scalar decay: one dimmer' : 'Channel-wise decay: a dimmer per column'}
          </button>
        ))}
      </div>

      <div className="intro-board__frame">
        <div className="intro-board__grid" role="img" aria-label="An 8 by 8 board of ink cells that fades over time">
          {Array.from({ length: N * N }, (_, i) => (
            <div
              key={i}
              className="intro-board__cell"
              ref={(el) => {
                cells.current[i] = el;
              }}
            />
          ))}
        </div>

        <motion.div
          key={mode}
          className={`intro-board__dimmers intro-board__dimmers--${mode === 'scalar' ? 'one' : 'per-column'}`}
          initial={reduce ? false : { opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: reduce ? 0 : 0.45 }}
        >
          {Array.from({ length: dimmerCount }, (_, i) => (
            <div key={i} className="intro-dimmer" style={{ borderColor: colour }}>
              <div
                className="intro-dimmer__fill"
                style={{ background: colour }}
                ref={(el) => {
                  dimmers.current[i] = el;
                }}
              />
            </div>
          ))}
        </motion.div>
        <p className="intro-board__label">
          {mode === 'scalar' ? 'One dimmer fades the whole board at the same rate.' : 'Each column has its own dimmer and rate.'}
        </p>
      </div>

      <WhatThisShows>
        {mode === 'scalar'
          ? 'every cell fades together, because one number sets the fade for the whole memory.'
          : 'columns fade at different speeds, because each column has its own number. The memory size is the same either way.'}
      </WhatThisShows>
    </div>
  );
}

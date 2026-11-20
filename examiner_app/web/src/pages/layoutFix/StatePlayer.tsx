import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import type { TryResponse } from '../../api';
import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Heatmap } from '../../components/Heatmap';
import { Term } from '../../components/Term';
import { TokenStrip } from '../../components/TokenStrip';
import { WhatThisShows } from '../../components/WhatThisShows';
import { maxAbs3 } from './derive';

/** Shared playback state. Pass one object to several players to scrub them together. */
export interface Playback {
  t: number;
  setT: (t: number) => void;
  playing: boolean;
  setPlaying: (p: boolean) => void;
  length: number;
}

/** Play/pause/scrub state for a sequence of `length` token positions. */
export function usePlayback(length: number): Playback {
  const [raw, setT] = useState(0);
  const [playing, setPlaying] = useState(false);
  const reduce = useReducedMotion() ?? false;
  const t = Math.min(raw, Math.max(0, length - 1));

  useEffect(() => {
    if (!playing || length === 0) return;
    const id = window.setInterval(() => setT((p) => Math.min(p + 1, length - 1)), reduce ? 500 : 280);
    return () => window.clearInterval(id);
  }, [playing, length, reduce]);

  useEffect(() => {
    if (playing && t >= length - 1) setPlaying(false);
  }, [playing, t, length]);

  return { t, setT, playing, setPlaying, length };
}

export function PlaybackControls({ pb }: { pb: Playback }) {
  const last = Math.max(0, pb.length - 1);
  const atEnd = pb.t >= last;
  return (
    <div className="lf-playback">
      <button type="button" className="lf-btn" aria-label="previous token" disabled={pb.length === 0 || pb.t === 0}
        onClick={() => { pb.setPlaying(false); pb.setT(Math.max(0, pb.t - 1)); }}>
        &#8249;
      </button>
      <button
        type="button"
        className="lf-btn lf-btn--primary"
        disabled={pb.length === 0}
        onClick={() => {
          if (pb.playing) {
            pb.setPlaying(false);
          } else {
            if (atEnd) pb.setT(0);
            pb.setPlaying(true);
          }
        }}
      >
        {pb.playing ? 'Pause' : 'Play'}
      </button>
      <button type="button" className="lf-btn" aria-label="next token" disabled={pb.length === 0 || atEnd}
        onClick={() => { pb.setPlaying(false); pb.setT(Math.min(last, pb.t + 1)); }}>
        &#8250;
      </button>
      <input
        className="lf-range"
        type="range"
        min={0}
        max={last}
        value={pb.t}
        aria-label="scrub through token positions"
        onChange={(e) => {
          pb.setPlaying(false);
          pb.setT(Number(e.target.value));
        }}
      />
      <span className="lf-playback__pos">
        position {pb.t} of {last}
      </span>
    </div>
  );
}

/** Line of the state norm after each token for one layer, drawn in on first view. */
function Sparkline({ ys, t, color, label }: { ys: number[]; t: number; color: string; label: string }) {
  const reduce = useReducedMotion() ?? false;
  const ref = useRef<SVGSVGElement>(null);
  const inView = useInView(ref, { once: true, amount: 0.3 });
  const W = 150;
  const H = 40;
  const hi = Math.max(1e-9, ...ys);
  const xAt = (i: number) => (ys.length <= 1 ? 0 : (i / (ys.length - 1)) * W);
  const yAt = (v: number) => H - 3 - (v / hi) * (H - 6);
  const d = ys.map((v, i) => `${i === 0 ? 'M' : 'L'}${xAt(i).toFixed(1)},${yAt(v).toFixed(1)}`).join(' ');
  const cur = ys[Math.min(t, ys.length - 1)] ?? 0;
  return (
    <svg ref={ref} className="lf-spark" viewBox={`0 0 ${W} ${H}`} width={W} height={H} role="img" aria-label={label}>
      <motion.path
        d={d}
        fill="none"
        stroke={color}
        strokeWidth={1.6}
        strokeLinejoin="round"
        initial={reduce ? false : { pathLength: 0 }}
        animate={{ pathLength: inView || reduce ? 1 : 0 }}
        transition={{ duration: reduce ? 0 : 1.4, ease: 'easeInOut' }}
      />
      <circle cx={xAt(Math.min(t, ys.length - 1))} cy={yAt(cur)} r={2.6} fill={color} />
    </svg>
  );
}

export interface StatePlayerProps {
  response: TryResponse;
  /** Heading, e.g. "A0 control · scalar · seed 42". */
  label: string;
  /** Arm colour used for the decay-gate bars and sparklines. */
  color: string;
  /** Shared playback; when omitted the player owns its own controls. */
  playback?: Playback;
  /** memory: token strip, memory grid and gates. size: state-size sparklines only. all: both (default). */
  view?: 'all' | 'memory' | 'size';
  /** Smaller cells and bars, for side-by-side players. */
  compact?: boolean;
  /** Show the token strip inside the player (set false when the strip sits in its own tab). Default true. */
  showTokens?: boolean;
}

/** Steps through one example token by token: token strip, layer-4 state crop, per-layer gates and state size. */
export function StatePlayer({ response, label, color, playback, view = 'all', compact = false, showTokens = true }: StatePlayerProps) {
  const showMemory = view !== 'size';
  const showSize = view !== 'memory';
  const len = response.example.tokens.length;
  const own = usePlayback(len);
  const pb = playback ?? own;
  const t = Math.min(pb.t, Math.max(0, len - 1));
  const { state_crops: crops, layers } = response.internals;

  const cropMax = useMemo(() => maxAbs3(crops), [crops]);
  const crop = crops[t] ?? [];
  const scalar = (layers[0]?.alpha[0]?.length ?? 1) === 1;

  const alphaBars: BarDatum[] = layers.map((L) => ({
    label: `Layer ${L.layer}`,
    value: L.alpha[t]?.[0] ?? 0,
    color,
  }));
  const betaBars: BarDatum[] = layers.map((L) => ({
    label: `Layer ${L.layer}`,
    value: L.beta[t] ?? 0,
    color: 'var(--text)',
  }));

  const selectAt = (i: number) => {
    pb.setPlaying(false);
    pb.setT(i);
  };

  return (
    <section className={`lf-player${compact ? ' lf-player--compact' : ''}`} aria-label={label}>
      <header className="lf-player__head">
        <span className="lf-swatch" style={{ background: color }} />
        <strong>{label}</strong>
      </header>
      {!playback && view === 'all' && <PlaybackControls pb={own} />}
      {showMemory && showTokens && (
        <TokenStrip tokens={response.example.tokens} roles={response.example.roles} selected={t} onSelect={selectAt} />
      )}

      {showMemory && (
        <div className="lf-player__grid">
          <div className="lf-player__state">
            <Heatmap
              matrix={crop}
              mode="diverging"
              max={cropMax}
              cellSize={compact ? 4 : 6}
              title={`Memory, layer 4 (32×32 corner), token ${t}`}
            />
          </div>

          {/* Scalar gates are one number per layer, so the two gate groups sit side by side. Channel-wise gates are stacked. */}
          <div className={scalar ? 'lf-player__gates lf-player__gates--pair' : 'lf-player__gates'}>
            <div className="lf-player__gatecol">
              <div className="lf-player__sub">
                <Term term="decay-gate" /> {scalar ? '(one number per layer)' : '(128 numbers per layer)'}
              </div>
              {scalar ? (
                <AnimatedBars data={alphaBars} max={1} format={(v) => v.toFixed(3)} />
              ) : (
                layers.map((L) => (
                  <Heatmap
                    key={`alpha-${L.layer}`}
                    matrix={[L.alpha[t] ?? []]}
                    mode="sequential"
                    max={1}
                    cellSize={3}
                    title={`Layer ${L.layer} α across 128 channels`}
                  />
                ))
              )}
            </div>
            <div className="lf-player__gatecol">
              <div className="lf-player__sub">
                <Term term="write-gate" />
              </div>
              <AnimatedBars data={betaBars} max={1} format={(v) => v.toFixed(3)} />
            </div>
          </div>
        </div>
      )}

      {showSize && (
        <>
          <div className="lf-player__sub">
            <Term term="recurrent-state" /> size after each token
          </div>
          <div className="lf-spark-row">
            {layers.map((L) => (
              <figure key={`spark-${L.layer}`} className="lf-spark-fig">
                <Sparkline
                  ys={L.state_norm}
                  t={t}
                  color={color}
                  label={`Layer ${L.layer} state size over the sequence, currently at position ${t}`}
                />
                <figcaption>Layer {L.layer}: {(L.state_norm[t] ?? 0).toFixed(2)}</figcaption>
              </figure>
            ))}
          </div>
        </>
      )}
      {view === 'all' && (
        <WhatThisShows>the grid is the memory after this token. Gate bars show how much old memory is kept and how strongly it is written.</WhatThisShows>
      )}
    </section>
  );
}

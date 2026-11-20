import { useEffect, useMemo, useRef, useState, type MouseEvent } from 'react';
import { useReducedMotion } from 'motion/react';
import { parseRgb, useThemeTokens, type RGB } from '../theme';
import './viz.css';

const DURATION_MS = 250;

export interface HeatmapProps {
  /** Rows of numbers, indexed [row][col]. A change of values tweens over 250 ms. */
  matrix: number[][];
  /** Screen pixels per cell (default 14). */
  cellSize?: number;
  /** diverging: neg - mid - pos colours (theme tokens), symmetric about 0. sequential: lo (0) to hi (1). */
  mode: 'diverging' | 'sequential';
  /** Value at the strongest colour (diverging: +/-max; sequential: max = light end). Default: largest |value| or 1. */
  max?: number;
  /** Heading shown above the canvas. */
  title?: string;
  /** Called while hovering a cell with its row, column and value. */
  onHover?: (row: number, col: number, value: number) => void;
}

// Colour stops come from the theme tokens (theme.css). Read as RGB because canvas cannot use var().
const HEAT_TOKENS = ['--heat-neg', '--heat-mid', '--heat-pos', '--heat-seq-lo', '--heat-seq-hi'] as const;
interface Palette {
  neg: RGB;
  mid: RGB;
  pos: RGB;
  lo: RGB;
  hi: RGB;
}

const clamp = (v: number, lo: number, hi: number) => Math.min(hi, Math.max(lo, v));
const lerp = (a: RGB, b: RGB, t: number): RGB => [
  a[0] + (b[0] - a[0]) * t,
  a[1] + (b[1] - a[1]) * t,
  a[2] + (b[2] - a[2]) * t,
];

function colourOf(v: number, mode: HeatmapProps['mode'], max: number, pal: Palette): RGB {
  const value = Number.isFinite(v) ? v : 0;
  if (mode === 'diverging') {
    const t = clamp(value / max, -1, 1);
    return t < 0 ? lerp(pal.mid, pal.neg, -t) : lerp(pal.mid, pal.pos, t);
  }
  return lerp(pal.lo, pal.hi, clamp(value / max, 0, 1));
}

const cssOf = (c: RGB) => `rgb(${Math.round(c[0])},${Math.round(c[1])},${Math.round(c[2])})`;

function paint(
  canvas: HTMLCanvasElement,
  cols: number,
  rows: number,
  values: Float64Array,
  mode: HeatmapProps['mode'],
  max: number,
  pal: Palette,
) {
  const ctx = canvas.getContext('2d');
  if (!ctx || cols === 0 || rows === 0) return;
  const img = ctx.createImageData(cols, rows);
  for (let i = 0; i < values.length; i++) {
    const [r, g, b] = colourOf(values[i], mode, max, pal);
    img.data[i * 4] = r;
    img.data[i * 4 + 1] = g;
    img.data[i * 4 + 2] = b;
    img.data[i * 4 + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
}

/** Canvas heatmap with diverging or sequential colour maps; tweens between matrices over 250 ms. */
export function Heatmap({ matrix, cellSize = 14, mode, max, title, onHover }: HeatmapProps) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const shownRef = useRef<Float64Array | null>(null);
  const reduce = useReducedMotion() ?? false;
  const tokens = useThemeTokens(HEAT_TOKENS);
  const pal = useMemo<Palette>(
    () => ({
      neg: parseRgb(tokens['--heat-neg']),
      mid: parseRgb(tokens['--heat-mid']),
      pos: parseRgb(tokens['--heat-pos']),
      lo: parseRgb(tokens['--heat-seq-lo']),
      hi: parseRgb(tokens['--heat-seq-hi']),
    }),
    [tokens],
  );
  const [readout, setReadout] = useState<{ row: number; col: number; value: number } | null>(null);

  const rows = matrix.length;
  const cols = matrix[0]?.length ?? 0;
  // Content key so that a parent re-rendering with identical numbers does not restart the tween.
  const dataKey = matrix.map((r) => r.join(',')).join(';');
  const flat = useMemo(() => Float64Array.from(matrix.flat()), [dataKey]);

  let absMax = 0;
  for (const v of flat) if (Number.isFinite(v)) absMax = Math.max(absMax, Math.abs(v));
  const colorMax = max ?? (mode === 'diverging' ? absMax || 1 : 1);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const prev = shownRef.current;
    if (reduce || !prev || prev.length !== flat.length) {
      paint(canvas, cols, rows, flat, mode, colorMax, pal);
      shownRef.current = flat;
      return;
    }
    const start = performance.now();
    let frame = 0;
    const tick = (now: number) => {
      const t = Math.min(1, (now - start) / DURATION_MS);
      const e = 1 - (1 - t) ** 3;
      const cur = new Float64Array(flat.length);
      for (let i = 0; i < flat.length; i++) cur[i] = prev[i] + (flat[i] - prev[i]) * e;
      shownRef.current = t >= 1 ? flat : cur;
      paint(canvas, cols, rows, cur, mode, colorMax, pal);
      if (t < 1) frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    // Interrupted tweens resume from whatever was last painted.
    return () => cancelAnimationFrame(frame);
  }, [flat, mode, colorMax, reduce, rows, cols, pal]);

  const onMove = (e: MouseEvent<HTMLCanvasElement>) => {
    const rect = e.currentTarget.getBoundingClientRect();
    if (rect.width === 0 || rect.height === 0) return;
    const col = Math.floor(((e.clientX - rect.left) / rect.width) * cols);
    const row = Math.floor(((e.clientY - rect.top) / rect.height) * rows);
    if (row < 0 || col < 0 || row >= rows || col >= cols) return;
    const value = matrix[row][col];
    setReadout({ row, col, value });
    onHover?.(row, col, value);
  };

  const legendStops = Array.from({ length: 9 }, (_, i) => {
    const t = i / 8;
    const v = mode === 'diverging' ? -colorMax + 2 * colorMax * t : colorMax * t;
    return cssOf(colourOf(v, mode, colorMax, pal));
  });

  return (
    <div className="viz-heatmap">
      {title && <div className="viz-heatmap__title">{title}</div>}
      <canvas
        ref={canvasRef}
        className="viz-heatmap__canvas"
        width={cols}
        height={rows}
        style={{ width: cols * cellSize, height: rows * cellSize }}
        onMouseMove={onMove}
        onMouseLeave={() => setReadout(null)}
      />
      <div className="viz-heatmap__readout" aria-live="polite">
        {readout
          ? `row ${readout.row}, col ${readout.col}: ${Number.isInteger(readout.value) ? readout.value : readout.value.toPrecision(3)}`
          : ' '}
      </div>
      <div className="viz-heatmap__legend">
        <span>{mode === 'diverging' ? `-${colorMax.toPrecision(2)}` : '0'}</span>
        <div className="viz-heatmap__bar" style={{ background: `linear-gradient(to right, ${legendStops.join(',')})` }} />
        <span>{mode === 'diverging' ? `+${colorMax.toPrecision(2)}` : colorMax.toPrecision(2)}</span>
      </div>
    </div>
  );
}

import { useEffect, useMemo, useRef, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';

export interface MiniSeries {
  name: string;
  ys: number[];
  color: string;
  dashed?: boolean;
}

/**
 * Small line chart with a fixed height, so it fits a slide. Responsive in width, draws each line in on first view,
 * optional log y axis (values must be positive). Used where the shared AnimatedLines (fixed 300 px) would overflow.
 */
export function MiniLines({
  series,
  xs,
  yLabel,
  xLabel,
  height = 170,
  log = false,
}: {
  series: MiniSeries[];
  xs: number[];
  yLabel: string;
  xLabel: string;
  height?: number;
  log?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(480);
  const reduce = useReducedMotion() ?? false;

  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const observer = new ResizeObserver(([entry]) => setWidth(Math.max(240, Math.floor(entry.contentRect.width))));
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  const geometry = useMemo(() => {
    const m = { l: 56, r: 10, t: 8, b: 30 };
    const f = (v: number) => (log ? Math.log10(v) : v);
    const all = series.flatMap((s) => s.ys).map(f).filter((v) => Number.isFinite(v));
    let lo = Math.min(...all);
    let hi = Math.max(...all);
    if (hi - lo < 1e-9) {
      lo -= 1;
      hi += 1;
    }
    const pad = (hi - lo) * 0.06;
    lo -= pad;
    hi += pad;
    const xMin = Math.min(...xs);
    const xMax = Math.max(...xs);
    const innerW = width - m.l - m.r;
    const innerH = height - m.t - m.b;
    const x = (v: number) => m.l + ((v - xMin) / (xMax - xMin || 1)) * innerW;
    const y = (v: number) => m.t + (1 - (f(v) - lo) / (hi - lo)) * innerH;
    const yTicks: { v: number; label: string }[] = [];
    if (log) {
      for (let p = Math.ceil(lo); p <= Math.floor(hi); p += 1) {
        const v = 10 ** p;
        yTicks.push({ v, label: v >= 1 ? String(Math.round(v)) : v.toFixed(2) });
      }
    } else {
      for (let i = 0; i <= 3; i += 1) {
        const v = lo + ((hi - lo) * i) / 3;
        yTicks.push({ v, label: v.toFixed(2) });
      }
    }
    const xTicks = [0, 1, 2, 3, 4].map((i) => xMin + ((xMax - xMin) * i) / 4);
    return { m, innerW, innerH, x, y, yTicks, xTicks, lo, hi, f };
  }, [series, xs, width, height, log]);

  return (
    <div className="di-chart" ref={ref}>
      <svg width={width} height={height} role="img" aria-label={`${yLabel} against ${xLabel}`}>
        {geometry.yTicks
          .filter((t) => geometry.f(t.v) >= geometry.lo && geometry.f(t.v) <= geometry.hi)
          .map((t) => (
            <g key={`y${t.label}${t.v}`}>
              <line x1={geometry.m.l} x2={width - geometry.m.r} y1={geometry.y(t.v)} y2={geometry.y(t.v)} className="di-chart__grid" />
              <text x={geometry.m.l - 6} y={geometry.y(t.v) + 4} textAnchor="end" className="di-chart__tick">
                {t.label}
              </text>
            </g>
          ))}
        {geometry.xTicks.map((v) => (
          <text key={`x${v}`} x={geometry.x(v)} y={height - 16} textAnchor="middle" className="di-chart__tick">
            {Math.round(v)}
          </text>
        ))}
        <text x={geometry.m.l + geometry.innerW / 2} y={height - 2} textAnchor="middle" className="di-chart__axis">
          {xLabel}
        </text>
        <text transform={`translate(12 ${geometry.m.t + geometry.innerH / 2}) rotate(-90)`} textAnchor="middle" className="di-chart__axis">
          {yLabel}
        </text>
        {series.map((s, i) => {
          const d = s.ys
            .map((v, j) => `${j === 0 ? 'M' : 'L'}${geometry.x(xs[j]).toFixed(1)},${geometry.y(v).toFixed(1)}`)
            .join(' ');
          return (
            <motion.path
              key={`${s.name}${i}`}
              d={d}
              fill="none"
              style={{ stroke: s.color }}
              strokeWidth={2}
              strokeDasharray={s.dashed ? '5 4' : undefined}
              initial={reduce ? false : { pathLength: 0 }}
              animate={{ pathLength: 1 }}
              transition={{ duration: 0.9, ease: 'easeOut' }}
            />
          );
        })}
      </svg>
      <div className="di-chart__legend">
        {series.map((s) => (
          <span key={s.name}>
            <i style={{ background: s.dashed ? 'none' : s.color, borderTop: s.dashed ? `2px dashed ${s.color}` : undefined }} />
            {s.name}
          </span>
        ))}
      </div>
    </div>
  );
}

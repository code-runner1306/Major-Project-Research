import { useEffect, useId, useRef, useState, type MouseEvent } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { format, line, scaleLinear, scaleLog, type ScaleContinuousNumeric } from 'd3';
import './viz.css';

const PALETTE = ['var(--accent)', 'var(--text-muted)', 'var(--c-good)', 'var(--tok-target-value)', 'var(--tok-query)'];
const HEIGHT = 300;
const MARGIN = { top: 12, right: 16, bottom: 44, left: 60 };
const fmtValue = format('.4~g');

export interface LineSeries {
  name: string;
  ys: number[];
  color?: string;
  /** Dashed stroke; these series reveal left-to-right instead of pathLength drawing. */
  dashed?: boolean;
}

export interface AnimatedLinesProps {
  /** Line series sharing one x axis; hover shows every series value at the nearest x. */
  series: LineSeries[];
  /** Label for the vertical axis. */
  yLabel: string;
  /** Label for the horizontal axis. */
  xLabel: string;
  /** Explicit x positions for every point (default 0..n-1). */
  xs?: number[];
  /** Upper bound of the y axis (default: data maximum). */
  yMax?: number;
  /** Use a logarithmic y axis; non-positive values are skipped. */
  log?: boolean;
}

/** SVG line chart: series draw in on view, hover crosshair and tooltip, optional log y axis, legend. */
export function AnimatedLines({ series, yLabel, xLabel, xs, yMax, log = false }: AnimatedLinesProps) {
  const wrapRef = useRef<HTMLDivElement>(null);
  const inView = useInView(wrapRef, { once: true, amount: 0.25 });
  const reduce = useReducedMotion() ?? false;
  const uid = useId().replace(/[^A-Za-z0-9_-]/g, '');
  const [width, setWidth] = useState(640);
  const [hover, setHover] = useState<number | null>(null);

  useEffect(() => {
    const el = wrapRef.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(Math.max(280, Math.floor(w)));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const n = Math.max(0, ...series.map((s) => s.ys.length));
  const xValues = xs ?? Array.from({ length: n }, (_, i) => i);
  const innerW = Math.max(40, width - MARGIN.left - MARGIN.right);
  const innerH = HEIGHT - MARGIN.top - MARGIN.bottom;

  const xLo = xValues.length ? Math.min(...xValues) : 0;
  const xHiRaw = xValues.length ? Math.max(...xValues) : 1;
  const xHi = xHiRaw > xLo ? xHiRaw : xLo + 1;
  const x = scaleLinear().domain([xLo, xHi]).range([0, innerW]);

  const finite = series.flatMap((s) => s.ys).filter((v) => Number.isFinite(v));
  const dataMax = finite.length ? Math.max(...finite) : 1;
  const hi = yMax ?? (dataMax > 0 ? dataMax : 1);
  let y: ScaleContinuousNumeric<number, number>;
  if (log) {
    const positives = finite.filter((v) => v > 0);
    const lo = positives.length ? Math.min(...positives) : 1e-3;
    y = scaleLog()
      .domain([lo * 0.8, Math.max(hi, lo * 2)])
      .range([innerH, 0])
      .nice();
  } else {
    y = scaleLinear()
      .domain([Math.min(0, ...finite), hi])
      .range([innerH, 0])
      .nice();
  }

  const yTicks = y.ticks(5);
  const yFmt = y.tickFormat(5);
  const xTicks = x.ticks(Math.min(8, Math.max(2, Math.floor(innerW / 90))));
  const xFmt = x.tickFormat(Math.min(8, Math.max(2, Math.floor(innerW / 90))));

  const gen = line<[number, number]>()
    .x((d) => x(d[0]))
    .y((d) => y(d[1]))
    .defined((d) => !log || d[1] > 0);

  const onMove = (e: MouseEvent<SVGRectElement>) => {
    if (n === 0) return;
    const px = e.clientX - e.currentTarget.getBoundingClientRect().left;
    const xv = x.invert(px);
    let best = 0;
    let bestDist = Infinity;
    xValues.forEach((v, i) => {
      const dist = Math.abs(v - xv);
      if (dist < bestDist) {
        bestDist = dist;
        best = i;
      }
    });
    setHover(best);
  };

  const hoverX = hover !== null && xValues[hover] !== undefined ? x(xValues[hover]) : null;
  const tooltipRight = hoverX !== null && hoverX > innerW / 2;

  return (
    <div className="viz-lines" ref={wrapRef}>
      <svg
        className="viz-lines__svg"
        width={width}
        height={HEIGHT}
        role="img"
        aria-label={`${yLabel} against ${xLabel}`}
      >
        <defs>
          <clipPath id={`${uid}-plot`}>
            <rect width={innerW} height={innerH} />
          </clipPath>
          {series.map((s, i) =>
            s.dashed ? (
              <clipPath id={`${uid}-reveal-${i}`} key={`reveal-${i}`}>
                <motion.rect
                  height={innerH}
                  initial={reduce ? false : { width: 0 }}
                  animate={{ width: inView || reduce ? innerW : 0 }}
                  transition={{ duration: reduce ? 0 : 1.6, ease: 'easeInOut', delay: reduce ? 0 : i * 0.15 }}
                />
              </clipPath>
            ) : null,
          )}
        </defs>

        <g transform={`translate(${MARGIN.left},${MARGIN.top})`}>
          {yTicks.map((t) => (
            <line key={`yg-${t}`} className="viz-lines__grid" x1={0} x2={innerW} y1={y(t)} y2={y(t)} />
          ))}
          <line className="viz-lines__axis" x1={0} x2={innerW} y1={innerH} y2={innerH} />
          <line className="viz-lines__axis" x1={0} x2={0} y1={0} y2={innerH} />
          {yTicks.map((t) => (
            <text key={`yt-${t}`} className="viz-lines__tick" x={-8} y={y(t)} dy="0.32em" textAnchor="end">
              {yFmt(t)}
            </text>
          ))}
          {xTicks.map((t) => (
            <text key={`xt-${t}`} className="viz-lines__tick" x={x(t)} y={innerH + 16} textAnchor="middle">
              {xFmt(t)}
            </text>
          ))}
          <text
            className="viz-lines__axis-label"
            transform={`translate(${-MARGIN.left + 14},${innerH / 2}) rotate(-90)`}
            textAnchor="middle"
          >
            {yLabel}
          </text>
          <text className="viz-lines__axis-label" x={innerW / 2} y={innerH + 38} textAnchor="middle">
            {xLabel}
          </text>

          <g clipPath={`url(#${uid}-plot)`}>
            {series.map((s, i) => {
              const color = s.color ?? PALETTE[i % PALETTE.length];
              const d = gen(s.ys.map((v, j) => [xValues[j] ?? j, v] as [number, number])) ?? '';
              if (s.dashed) {
                // motion's pathLength owns stroke-dasharray, so dashed series reveal through a clip instead.
                return (
                  <g key={`s-${i}`} clipPath={`url(#${uid}-reveal-${i})`}>
                    <path d={d} fill="none" style={{ stroke: color }} strokeWidth={2} strokeDasharray="6 5" />
                  </g>
                );
              }
              return (
                <motion.path
                  key={`s-${i}`}
                  d={d}
                  fill="none"
                  style={{ stroke: color }}
                  strokeWidth={2.25}
                  strokeLinejoin="round"
                  initial={reduce ? false : { pathLength: 0 }}
                  animate={{ pathLength: inView || reduce ? 1 : 0 }}
                  transition={{ duration: reduce ? 0 : 1.6, ease: 'easeInOut', delay: reduce ? 0 : i * 0.15 }}
                />
              );
            })}
          </g>

          {hoverX !== null && (
            <line className="viz-lines__cross" x1={hoverX} x2={hoverX} y1={0} y2={innerH} />
          )}
          {hover !== null &&
            hoverX !== null &&
            series.map((s, i) => {
              const v = s.ys[hover];
              if (v === undefined || !Number.isFinite(v) || (log && v <= 0)) return null;
              return (
                <circle
                  key={`dot-${i}`}
                  cx={hoverX}
                  cy={y(v)}
                  r={4}
                  style={{ fill: 'var(--surface)', stroke: s.color ?? PALETTE[i % PALETTE.length] }}
                  strokeWidth={2}
                  pointerEvents="none"
                />
              );
            })}

          <rect
            width={innerW}
            height={innerH}
            fill="transparent"
            onMouseMove={onMove}
            onMouseLeave={() => setHover(null)}
          />
        </g>
      </svg>

      {hover !== null && hoverX !== null && (
        <div
          className="viz-lines__tooltip"
          style={{
            left: MARGIN.left + hoverX + (tooltipRight ? -10 : 10),
            transform: tooltipRight ? 'translateX(-100%)' : undefined,
          }}
        >
          <div className="viz-lines__tooltip-x">
            {xLabel}: {fmtValue(xValues[hover])}
          </div>
          {series.map((s, i) => {
            const v = s.ys[hover];
            const color = s.color ?? PALETTE[i % PALETTE.length];
            return (
              <div className="viz-lines__tooltip-row" key={`tt-${i}`} style={{ color }}>
                <span className={`viz-lines__swatch${s.dashed ? ' viz-lines__swatch--dashed' : ''}`} />
                <span style={{ color: 'var(--text)' }}>{s.name}</span>
                <span className="viz-lines__tooltip-val" style={{ color: 'var(--text)' }}>
                  {v === undefined || !Number.isFinite(v) ? '–' : fmtValue(v)}
                </span>
              </div>
            );
          })}
        </div>
      )}

      <ul className="viz-lines__legend">
        {series.map((s, i) => (
          <li key={`lg-${i}`} style={{ color: s.color ?? PALETTE[i % PALETTE.length] }}>
            <span className={`viz-lines__swatch${s.dashed ? ' viz-lines__swatch--dashed' : ''}`} />
            <span style={{ color: 'var(--text)' }}>{s.name}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

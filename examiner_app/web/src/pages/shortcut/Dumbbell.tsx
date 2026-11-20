import { useRef, useState } from 'react';
import { motion, useInView, useReducedMotion } from 'motion/react';
import { scaleLinear } from 'd3';
import { WhatThisShows } from '../../components/WhatThisShows';
import { useElementSize } from '../../components/useElementSize';
import { pct, variantText } from './common';
import { initLabel, type PerModel } from './types';

type Plotted = PerModel & { block_first_r3: number; general_r3: number };

// Geometry in CSS pixels (the SVG is drawn at its measured size, so text never shrinks).
const VALUE_W = 150; // right-hand column: "block-first / general" values, never on a line
const ROW_MIN = 22; // smallest row height that keeps a row readable
const ROW_MAX = 30;
const HEAD = 26; // column headings
const AXIS = 28; // tick labels
const PLOT_PAD = 28; // gap between the label column and the plot
const DOT_R = 6;

// Short label for one model, e.g. "scalar · default · seed 42". Its width sets the label column.
function labelOf(r: PerModel): string {
  return `${variantText(r.variant)} · ${initLabel(r.init).replace(' init', '')} · seed ${r.seed}`;
}

let measureCtx: CanvasRenderingContext2D | null | undefined;
/** Width in pixels of `text` in the page's mono font at the row-label size. */
function labelWidth(text: string): number {
  if (measureCtx === undefined) measureCtx = document.createElement('canvas').getContext('2d');
  if (!measureCtx) return text.length * 7.4;
  const family = getComputedStyle(document.documentElement).getPropertyValue('--font-mono').trim() || 'monospace';
  measureCtx.font = `12px ${family}`;
  return measureCtx.measureText(text).width;
}

/**
 * Per model: block-first and general accuracy at 3 rewrites as a dumbbell. Model labels sit in their own left column,
 * the plot starts after a gap, and the two values sit in a right-hand column. Shows 12 local models by default
 * when all 18 would not fit at ROW_MIN; a toggle shows the rest.
 */
export function Dumbbell({ rows }: { rows: PerModel[] }) {
  const wrap = useRef<HTMLDivElement>(null);
  const inView = useInView(wrap, { once: true, amount: 0.25 });
  const reduce = useReducedMotion() ?? false;
  const shown = inView || reduce;
  const [stageRef, box] = useElementSize<HTMLDivElement>();
  const [showAllPref, setShowAllPref] = useState<boolean | null>(null);

  const all = rows
    .filter((r): r is Plotted => r.block_first_r3 !== null && r.general_r3 !== null)
    .sort((a, b) => Number(b.local) - Number(a.local) || a.block_first_r3 - b.block_first_r3);

  if (all.length === 0) {
    return <p className="sc-muted">No model has both block-first and general results saved.</p>;
  }

  const local = all.filter((r) => r.local);
  const hiddenCount = all.length - local.length;
  const fitsAll = box.height === 0 || box.height >= HEAD + AXIS + all.length * ROW_MIN;
  const showAll = showAllPref ?? fitsAll;
  const plotted = showAll || local.length === 0 ? all : local;

  const W = box.width;
  const H = box.height;
  const n = plotted.length;
  const rowH = H > 0 ? Math.max(ROW_MIN, Math.min(ROW_MAX, (H - HEAD - AXIS) / n)) : ROW_MIN;
  const svgH = Math.max(H, HEAD + AXIS + n * rowH);

  const labelCol = Math.ceil(Math.max(0, ...plotted.map((r) => labelWidth(labelOf(r)))));
  const X0 = labelCol + PLOT_PAD;
  const valueX = Math.max(X0 + 160, W - VALUE_W);
  const X1 = valueX - 24;

  const lowest = Math.min(...plotted.flatMap((r) => [r.block_first_r3, r.general_r3]));
  const x = scaleLinear()
    .domain([Math.max(0, Math.floor((lowest - 0.05) * 10) / 10), 1])
    .range([X0, X1]);
  const ticks = x.ticks(W > 560 ? 5 : 3);
  const plotBottom = HEAD + n * rowH;

  return (
    <div className="sc-dumbbell" ref={wrap}>
      <div className="sc-dumbbell__bar">
        <ul className="sc-legend">
          <li>
            <span className="sc-swatch is-bf" /> block-first, 3 rewrites
          </li>
          <li>
            <span className="sc-swatch is-gen" /> general, 3 rewrites
          </li>
          <li>
            <span className="sc-swatch is-hollow" /> hollow: models unavailable here
          </li>
        </ul>
        {hiddenCount > 0 && (
          <button type="button" className="sc-btn sc-dumbbell__toggle" onClick={() => setShowAllPref(!showAll)}>
            {showAll ? `Show the ${local.length} local models` : `Show all ${all.length}`}
          </button>
        )}
      </div>

      <div className="sc-dumbbell__stage" ref={stageRef}>
        {W > 0 && H > 0 && (
          <svg
            width={W}
            height={svgH}
            className="sc-dumbbell__svg"
            role="img"
            aria-label="Block-first and general accuracy at three rewrites, per model"
          >
            <text x={valueX} y={14} className="sc-colhead">
              block-first / general
            </text>
            {ticks.map((t) => (
              <g key={t}>
                <line x1={x(t)} x2={x(t)} y1={HEAD - 6} y2={plotBottom} className="sc-grid" />
                <text x={x(t)} y={plotBottom + 18} className="sc-tick" textAnchor="middle">
                  {pct(t, 0)}
                </text>
              </g>
            ))}
            {plotted.map((r, i) => {
              const y = HEAD + i * rowH + rowH / 2;
              const xb = x(r.block_first_r3);
              const xg = x(r.general_r3);
              const mid = (xb + xg) / 2;
              const delay = reduce ? 0 : i * 0.04;
              const hollow = !r.local;
              return (
                <g key={`${r.init}-${r.variant}-${r.seed}`}>
                  <text x={0} y={y + 4} className={hollow ? 'sc-row-label is-unavailable' : 'sc-row-label'}>
                    {labelOf(r)}
                  </text>
                  <motion.line
                    x1={xg}
                    x2={xb}
                    y1={y}
                    y2={y}
                    className={hollow ? 'sc-link is-unavailable' : 'sc-link'}
                    initial={{ pathLength: 0 }}
                    animate={{ pathLength: shown ? 1 : 0 }}
                    transition={{ duration: reduce ? 0 : 0.6, delay }}
                  />
                  <motion.g
                    initial={false}
                    animate={{ x: shown ? xb - mid : 0 }}
                    transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 90, damping: 16, delay }}
                  >
                    <circle cx={mid} cy={y} r={DOT_R} className={hollow ? 'sc-dot is-bf is-hollow' : 'sc-dot is-bf'}>
                      <title>{`Block-first, 3 rewrites: ${pct(r.block_first_r3)}`}</title>
                    </circle>
                  </motion.g>
                  <motion.g
                    initial={false}
                    animate={{ x: shown ? xg - mid : 0 }}
                    transition={reduce ? { duration: 0 } : { type: 'spring', stiffness: 90, damping: 16, delay }}
                  >
                    <circle cx={mid} cy={y} r={DOT_R} className={hollow ? 'sc-dot is-gen is-hollow' : 'sc-dot is-gen'}>
                      <title>{`General, 3 rewrites: ${pct(r.general_r3)}`}</title>
                    </circle>
                  </motion.g>
                  {hollow ? (
                    <text x={valueX} y={y + 4} className="sc-marker">
                      Models unavailable
                    </text>
                  ) : (
                    <text x={valueX} y={y + 4} className="sc-values">
                      {`${pct(r.block_first_r3, 0)} / ${pct(r.general_r3, 0)}`}
                    </text>
                  )}
                </g>
              );
            })}
          </svg>
        )}
      </div>
      <WhatThisShows>each row is one model. The farther apart the dots, the stronger the shortcut.</WhatThisShows>
    </div>
  );
}

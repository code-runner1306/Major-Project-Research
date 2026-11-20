import { useEffect, useRef } from 'react';
import { readToken } from '../../theme';
import { WhatThisShows } from '../../components/WhatThisShows';
import type { AttentionControlSummary, OverwriteExample } from './types';

const CELL = 2.5; // CSS pixels per grid cell

/**
 * Causal attention grid for one real example. Row = the token doing the reading, column = a token it may read.
 * Only columns at or before the row are allowed (causal). Rows where the model answers are highlighted.
 */
function CausalGrid({ example }: { example: OverwriteExample }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const n = example.tokens.length;
  const size = n * CELL;

  useEffect(() => {
    const canvas = ref.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.round(size * dpr);
    canvas.height = Math.round(size * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, size, size);

    const allowed = readToken('--tok-filler') || '#d2d2d2';
    const answering = readToken('--accent') || '#e10600';
    const keyColour = readToken('--tok-target-key') || '#c9a46a';
    const answerRows = new Set(example.queries.map((q) => q.key_position));
    const keyCols = new Set(example.writes.key_positions);

    for (let row = 0; row < n; row += 1) {
      for (let col = 0; col <= row; col += 1) {
        ctx.fillStyle = answerRows.has(row) ? answering : allowed;
        ctx.fillRect(col * CELL, row * CELL, CELL - 0.5, CELL - 0.5);
      }
    }
    ctx.fillStyle = keyColour;
    for (const col of keyCols) ctx.fillRect(col * CELL, 0, CELL - 0.5, size);
  }, [example, n, size]);

  return (
    <canvas
      ref={ref}
      className="ac-grid"
      style={{ width: size, height: size }}
      role="img"
      aria-label={`Causal attention grid: each of ${n} positions can read itself and every earlier position`}
    />
  );
}

/** Left: attention. Right: recurrent memory as one fixed-size table, drawn from the saved backbone sizes. */
export function InsideVisual({
  example,
  summary,
}: {
  example: OverwriteExample;
  summary: AttentionControlSummary;
}) {
  const m = summary.memory;
  const n = example.tokens.length;
  return (
    <div className="ac-inside">
      <div className="ac-inside__col">
        <h3 className="ac-h3">Attention: each position reads every earlier position</h3>
        <CausalGrid example={example} />
        <p className="ac-note">Red rows answer the two questions; tan columns mark the queried key&apos;s writes.</p>
      </div>
      <div className="ac-inside__col">
        <h3 className="ac-h3">Recurrent memory: one fixed table</h3>
        <div className="ac-stream" aria-hidden="true" style={{ width: n * CELL }}>
          {Array.from({ length: n }, (_, i) => (
            <span key={i} className="ac-stream__tick" />
          ))}
        </div>
        <div className="ac-arrow" aria-hidden="true" />
        <div className="ac-table ac-table--big" role="img" aria-label={`${m.heads} tables of ${m.head_width} by ${m.head_width} numbers`}>
          {Array.from({ length: m.heads }, (_, i) => (
            <span key={i} className="ac-table__cell">
              {m.head_width}×{m.head_width}
            </span>
          ))}
        </div>
        <p className="ac-note">The same table is used at every position, however long the input.</p>
      </div>
      <WhatThisShows>schematic: attention weights were not saved, so this shows what can be read, not how much.</WhatThisShows>
    </div>
  );
}

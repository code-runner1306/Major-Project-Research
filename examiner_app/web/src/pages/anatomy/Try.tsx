import { useMemo, useState } from 'react';
import { Unavailable } from '../../components/Unavailable';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Remote } from '../shortcut/common';
import { useAnatomyRuns } from './data';
import { pct } from './Results';
import type { RunRow, Summary } from './types';

const PAGE_SIZE = 8;

/**
 * No checkpoints for these runs exist on this machine, so the try-it control is replaced by a filtered, paged view
 * of the saved runs (ours and the published reference arms). Eight rows per page.
 */
export function TryVisual({ summary }: { summary: Summary }) {
  const runs = useAnatomyRuns();
  const [cell, setCell] = useState<'all' | RunRow['cell']>('all');
  const [source, setSource] = useState<'all' | RunRow['source']>('all');
  const [page, setPage] = useState(0);
  const titleOf = useMemo(() => new Map(summary.cells.map((c) => [c.id, c.title])), [summary]);

  return (
    <div className="anat-try">
      <Unavailable reason="Models unavailable on this machine. Saved runs are listed below." />
      <Remote state={runs} what="the saved runs" tall>
        {(rows) => {
          const shown = rows.filter((r) => (cell === 'all' || r.cell === cell) && (source === 'all' || r.source === source));
          const pages = Math.max(1, Math.ceil(shown.length / PAGE_SIZE));
          const current = Math.min(page, pages - 1);
          const slice = shown.slice(current * PAGE_SIZE, current * PAGE_SIZE + PAGE_SIZE);
          const changeFilter = (fn: () => void) => {
            fn();
            setPage(0);
          };
          return (
            <div className="anat-try__body">
              <div className="anat-controls">
                <label className="anat-select">
                  <span>Experiment</span>
                  <select value={cell} onChange={(e) => changeFilter(() => setCell(e.target.value as 'all' | RunRow['cell']))}>
                    <option value="all">All three</option>
                    {summary.cells.map((c) => (
                      <option key={c.id} value={c.id}>
                        {c.title}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="anat-select">
                  <span>Runs</span>
                  <select value={source} onChange={(e) => changeFilter(() => setSource(e.target.value as 'all' | RunRow['source']))}>
                    <option value="all">Ours and published</option>
                    <option value="ours">Ours only</option>
                    <option value="published">Published only</option>
                  </select>
                </label>
              </div>
              <table className="anat-runs">
                <thead>
                  <tr>
                    <th scope="col">Experiment</th>
                    <th scope="col">Model</th>
                    <th scope="col">Seed</th>
                    <th scope="col">Accuracy</th>
                    <th scope="col">Rebinding control</th>
                    <th scope="col">Locked in</th>
                  </tr>
                </thead>
                <tbody>
                  {slice.map((r, i) => (
                    <tr key={`${current}-${i}`}>
                      <td>{titleOf.get(r.cell) ?? r.cell}</td>
                      <td>{r.label}</td>
                      <td>{r.seed}</td>
                      <td>{pct(r.eval_accuracy)}</td>
                      <td>{r.rebind_accuracy === null ? '—' : pct(r.rebind_accuracy)}</td>
                      <td>{r.lock_in ? 'yes' : 'no'}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
              <div className="anat-pager">
                <button type="button" disabled={current === 0} onClick={() => setPage(current - 1)}>
                  Previous
                </button>
                <span>
                  Page {current + 1} of {pages} · {shown.length} runs
                </span>
                <button type="button" disabled={current >= pages - 1} onClick={() => setPage(current + 1)}>
                  Next
                </button>
              </div>
              <WhatThisShows>the rebinding control falls to guessing when a model relies on the table&apos;s bindings.</WhatThisShows>
            </div>
          );
        }}
      </Remote>
    </div>
  );
}

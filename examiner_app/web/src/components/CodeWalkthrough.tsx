import { useMemo, useState, type KeyboardEvent } from 'react';
import type { CodeStep } from '../types';
import { useApi } from './useApi';
import './viz.css';

export interface CodeWalkthroughProps {
  /** Ordered steps: each shows a plain-English box above the real source lines it refers to. */
  steps: CodeStep[];
}

interface CodeResponse {
  path: string;
  symbol?: string | null;
  start_line: number;
  end_line: number;
  source: string;
}

type TokKind = 'plain' | 'kw' | 'str' | 'com' | 'num' | 'fn' | 'deco';
interface Tok {
  text: string;
  kind: TokKind;
}

const PY_KEYWORDS = new Set([
  'False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue',
  'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in',
  'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield',
]);

// Groups: 1 comment, 2 string, 3 "def"/"class " prefix, 4 defined name, 5 decorator, 6 number, 7 identifier.
const PY_PATTERN =
  '(#.*$)' +
  '|("""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|"(?:\\\\.|[^"\\\\])*"?|\'(?:\\\\.|[^\'\\\\])*\'?)' +
  '|(\\b(?:def|class)\\s+)([A-Za-z_]\\w*)' +
  '|(@[A-Za-z_][\\w.]*)' +
  '|(\\b0[xX][0-9a-fA-F_]+\\b|\\b\\d[\\d_]*(?:\\.\\d+)?(?:[eE][+-]?\\d+)?\\b)' +
  '|([A-Za-z_]\\w*)';

/** Dependency-free, line-at-a-time Python tokenizer for display colouring (multi-line strings are approximated). */
function tokenizePython(line: string): Tok[] {
  const out: Tok[] = [];
  const re = new RegExp(PY_PATTERN, 'g');
  let last = 0;
  let m: RegExpExecArray | null;
  while ((m = re.exec(line)) !== null) {
    if (m.index > last) out.push({ text: line.slice(last, m.index), kind: 'plain' });
    if (m[1] !== undefined) out.push({ text: m[1], kind: 'com' });
    else if (m[2] !== undefined) out.push({ text: m[2], kind: 'str' });
    else if (m[3] !== undefined) {
      out.push({ text: m[3], kind: 'kw' });
      out.push({ text: m[4], kind: 'fn' });
    } else if (m[5] !== undefined) out.push({ text: m[5], kind: 'deco' });
    else if (m[6] !== undefined) out.push({ text: m[6], kind: 'num' });
    else if (m[7] !== undefined) {
      out.push({ text: m[7], kind: PY_KEYWORDS.has(m[7]) ? 'kw' : 'plain' });
    }
    last = re.lastIndex;
    if (m[0].length === 0) re.lastIndex++;
  }
  if (last < line.length) out.push({ text: line.slice(last), kind: 'plain' });
  return out;
}

function codeUrl(step: CodeStep | undefined): string | null {
  if (!step) return null;
  const path = encodeURIComponent(step.path);
  if (step.symbol) return `/api/code?path=${path}&symbol=${encodeURIComponent(step.symbol)}`;
  if (step.lines) return `/api/code/lines?path=${path}&start=${step.lines.start}&end=${step.lines.end}`;
  return null;
}

/** Step-by-step walkthrough of real source: plain-English box, numbered code, and notes on highlighted lines. */
export function CodeWalkthrough({ steps }: CodeWalkthroughProps) {
  const [active, setActive] = useState(0);
  const [hovered, setHovered] = useState<number | null>(null);
  const [pinned, setPinned] = useState<number | null>(null);

  const step: CodeStep | undefined = steps[active];
  const url = codeUrl(step);
  const code = useApi<CodeResponse>(url);

  const lines = useMemo(
    () => (code.data ? code.data.source.replace(/\r?\n$/, '').split(/\r?\n/) : []),
    [code.data],
  );

  if (!step) return <p className="viz-code__empty">No walkthrough steps.</p>;

  const selectStep = (i: number) => {
    setActive(i);
    setHovered(null);
    setPinned(null);
  };

  const notesFor = (text: string) =>
    step.notes.filter((n) => n.match.length > 0 && text.includes(n.match));

  const focusLine = pinned ?? hovered;
  const focusNotes = focusLine !== null && lines[focusLine] !== undefined ? notesFor(lines[focusLine]) : [];

  const onTabKey = (e: KeyboardEvent<HTMLDivElement>) => {
    if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft') return;
    const delta = e.key === 'ArrowRight' ? 1 : -1;
    const next = (active + delta + steps.length) % steps.length;
    selectStep(next);
    const buttons = e.currentTarget.querySelectorAll<HTMLButtonElement>('[role="tab"]');
    buttons[next]?.focus();
  };

  return (
    <section className="viz-code">
      <div className="viz-code__tabs" role="tablist" onKeyDown={onTabKey}>
        {steps.map((s, i) => (
          <button
            key={`${i}:${s.title}`}
            type="button"
            role="tab"
            aria-selected={i === active}
            tabIndex={i === active ? 0 : -1}
            className="viz-code__tab"
            onClick={() => selectStep(i)}
          >
            {`${i + 1} ${s.title.replace(/^\d+\s+/, '')}`}
          </button>
        ))}
      </div>

      <p className="viz-code__plain">{step.plain}</p>

      <div className="viz-code__body">
        <div className="viz-code__main">
          <div className="viz-code__header">
            {code.data
              ? `${code.data.path}:${code.data.start_line}–${code.data.end_line}`
              : step.path}
          </div>

          {code.loading && <div className="viz-code__status">Loading code…</div>}
          {code.error && (
            <div className="viz-code__error" role="alert">
              Could not load code: {code.error}
            </div>
          )}
          {!url && !code.loading && <div className="viz-code__status">This step has no code range.</div>}

          {code.data && (
            <div className="viz-code__block">
              {lines.map((text, i) => {
                const matched = notesFor(text).length > 0;
                const isActive = matched && focusLine === i;
                const className = [
                  'viz-code__line',
                  matched ? 'viz-code__line--note' : '',
                  isActive ? 'viz-code__line--active' : '',
                ]
                  .filter(Boolean)
                  .join(' ');
                return (
                  <div
                    key={i}
                    className={className}
                    role={matched ? 'button' : undefined}
                    tabIndex={matched ? 0 : undefined}
                    aria-pressed={matched ? pinned === i : undefined}
                    onMouseEnter={() => matched && setHovered(i)}
                    onMouseLeave={() => setHovered(null)}
                    onClick={() => matched && setPinned((p) => (p === i ? null : i))}
                    onKeyDown={(e) => {
                      if (matched && (e.key === 'Enter' || e.key === ' ')) {
                        e.preventDefault();
                        setPinned((p) => (p === i ? null : i));
                      }
                    }}
                  >
                    <span className="viz-code__no">{code.data!.start_line + i}</span>
                    <span className="viz-code__text">
                      {tokenizePython(text).map((t, j) => (
                        <span key={j} className={t.kind === 'plain' ? undefined : `viz-code__${t.kind}`}>
                          {t.text}
                        </span>
                      ))}
                    </span>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <aside className="viz-code__notes" aria-live="polite">
          <h4>Notes</h4>
          {focusNotes.length > 0 && focusLine !== null ? (
            <>
              <p className="viz-code__empty" style={{ marginBottom: 8 }}>
                Line {code.data ? code.data.start_line + focusLine : focusLine + 1}
              </p>
              {focusNotes.map((n, k) => (
                <p className="viz-code__note" key={k}>
                  <code>{n.match}</code>
                  <br />
                  {n.note}
                </p>
              ))}
            </>
          ) : (
            <p className="viz-code__empty">Hover or click a highlighted line to see its note.</p>
          )}
        </aside>
      </div>
    </section>
  );
}

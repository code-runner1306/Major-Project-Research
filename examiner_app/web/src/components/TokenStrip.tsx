import { motion, useReducedMotion } from 'motion/react';
import './viz.css';

export const TOKEN_ROLES = [
  'filler',
  'key',
  'value',
  'target_key',
  'target_value',
  'marker',
  'query_latest',
  'query_unrelated',
  'query_key',
  'answer_slot',
  'noise',
  'decoy_key',
  'decoy_value',
  'mask',
  'text',
  'eos',
] as const;

const ROLE_LABEL: Record<string, string> = {
  text: 'story text',
  eos: 'end of text',
  filler: 'filler',
  key: 'key',
  value: 'value',
  target_key: 'target key',
  target_value: 'target value',
  marker: 'marker',
  query_latest: 'latest query',
  query_unrelated: 'unrelated query',
  query_key: 'query key',
  answer_slot: 'answer slot',
  noise: 'noise',
  decoy_key: 'decoy key',
  decoy_value: 'decoy value',
  mask: 'mask',
};

const MIN_RUN = 3; // runs strictly longer than this collapse when compactFiller is on

export interface TokenStripProps {
  /** Token ids (or any printable labels), one per position. */
  tokens: (number | string)[];
  /** Role per position, e.g. 'filler' | 'key' | 'value' | 'target_key' | 'target_value' | 'marker' | 'query_latest' | 'query_unrelated' | 'query_key' | 'answer_slot' | 'noise' | 'decoy_key' | 'decoy_value' | 'mask'. */
  roles: string[];
  /** Index of the selected position (outlined chip). */
  selected?: number;
  /** Called with a position index when a chip is activated (for a collapsed run, its first index). */
  onSelect?: (index: number) => void;
  /** Collapse runs of more than 3 filler tokens into a single "… n× filler" chip. */
  compactFiller?: boolean;
  /** Positions that are scored answers. They get an outline and are never hidden inside a collapsed filler run. */
  scored?: number[];
  /**
   * Decoded word piece per position (same length as tokens). When given, chips show the piece instead of the id;
   * the id stays in the chip's title. Spaces show as '·', newlines as '⏎', tabs as '⇥'. The eos role shows 'EOT'.
   */
  tokenText?: string[];
}

/** Makes a decoded piece visible: whitespace becomes a marker, an empty piece becomes '∅'. */
export function visiblePiece(piece: string): string {
  if (piece.length === 0) return '∅';
  return piece.replace(/ /g, '·').replace(/\r/g, '␍').replace(/\n/g, '⏎').replace(/\t/g, '⇥');
}

type Segment = { kind: 'token'; index: number } | { kind: 'run'; start: number; end: number };

function buildSegments(roles: string[], count: number, compactFiller: boolean, scored: Set<number>): Segment[] {
  const out: Segment[] = [];
  let i = 0;
  while (i < count) {
    if (compactFiller && roles[i] === 'filler' && !scored.has(i)) {
      let j = i;
      while (j + 1 < count && roles[j + 1] === 'filler' && !scored.has(j + 1)) j++;
      const len = j - i + 1;
      if (len > MIN_RUN) {
        out.push({ kind: 'run', start: i, end: j });
      } else {
        for (let k = i; k <= j; k++) out.push({ kind: 'token', index: k });
      }
      i = j + 1;
    } else {
      out.push({ kind: 'token', index: i });
      i++;
    }
  }
  return out;
}

/** Row of role-coloured token chips with optional collapsed filler runs; chips animate in with a short stagger. */
export function TokenStrip({ tokens, roles, selected, onSelect, compactFiller = false, scored, tokenText }: TokenStripProps) {
  const reduce = useReducedMotion() ?? false;
  const count = Math.min(tokens.length, roles.length);
  const scoredSet = new Set(scored ?? []);
  const segments = buildSegments(roles, count, compactFiller, scoredSet);
  const presentRoles = Array.from(new Set(roles.slice(0, count)));

  return (
    <div>
      <div className="viz-tokens">
        {segments.map((seg, s) => {
          const delay = reduce ? 0 : Math.min(s * 0.015, 0.6);
          if (seg.kind === 'run') {
            const n = seg.end - seg.start + 1;
            const isSel = selected !== undefined && selected >= seg.start && selected <= seg.end;
            return (
              <motion.button
                key={`run-${seg.start}`}
                type="button"
                className={`viz-token viz-token--filler-run${isSel ? ' viz-token--selected' : ''}`}
                initial={reduce ? false : { opacity: 0, y: 4 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.25, delay }}
                aria-label={`${n} filler tokens, positions ${seg.start} to ${seg.end}`}
                aria-pressed={onSelect ? isSel : undefined}
                title={`positions ${seg.start}–${seg.end}`}
                onClick={onSelect ? () => onSelect(seg.start) : undefined}
              >
                {`… ${n}× filler`}
              </motion.button>
            );
          }
          const i = seg.index;
          const role = roles[i];
          const isSel = selected === i;
          const isScored = scoredSet.has(i);
          const piece = tokenText?.[i];
          const text =
            role === 'filler' && !isScored
              ? '·'
              : role === 'marker' || role === 'answer_slot'
                ? '?'
                : role === 'mask'
                  ? '■'
                  : role === 'eos'
                    ? 'EOT'
                    : piece !== undefined
                      ? visiblePiece(piece)
                      : String(tokens[i]);
          const shownName = piece !== undefined && role !== 'eos' ? `"${piece}"` : `token ${String(tokens[i])}`;
          const classes = [
            'viz-token',
            `viz-token--${role}`,
            isSel ? 'viz-token--selected' : '',
            isScored ? 'viz-token--scored' : '',
          ]
            .filter(Boolean)
            .join(' ');
          return (
            <motion.button
              key={`tok-${i}`}
              type="button"
              className={classes}
              initial={reduce ? false : { opacity: 0, y: 4 }}
              animate={{ opacity: 1, y: 0 }}
              transition={{ duration: 0.25, delay }}
              aria-label={`${ROLE_LABEL[role] ?? role} ${shownName}, position ${i}${isScored ? ', scored answer' : ''}`}
              aria-pressed={onSelect ? isSel : undefined}
              title={`${ROLE_LABEL[role] ?? role} · position ${i} · id ${String(tokens[i])}${isScored ? ' · scored answer' : ''}`}
              onClick={onSelect ? () => onSelect(i) : undefined}
            >
              {text}
            </motion.button>
          );
        })}
      </div>
      <div className="viz-tokens__legend">
        {presentRoles.map((role) => (
          <span className="viz-tokens__legend-item" key={role}>
            <span className={`viz-tokens__swatch viz-token--${role}`} />
            {ROLE_LABEL[role] ?? role}
          </span>
        ))}
      </div>
    </div>
  );
}

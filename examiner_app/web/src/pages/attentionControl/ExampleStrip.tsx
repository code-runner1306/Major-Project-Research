import type { OverwriteExample } from './types';

// Role names in plain English, and the colour token each role uses (defined in theme.css).
const ROLE_INFO: Record<string, { label: string; color: string }> = {
  filler: { label: 'filler', color: 'var(--tok-filler)' },
  key: { label: 'other key', color: 'var(--tok-key)' },
  value: { label: 'other value', color: 'var(--tok-value)' },
  target_key: { label: 'queried key', color: 'var(--tok-target-key)' },
  target_value: { label: "queried key's value", color: 'var(--tok-target-value)' },
  marker: { label: 'question starts', color: 'var(--tok-marker)' },
  query_latest: { label: 'asked: queried key', color: 'var(--tok-query)' },
  query_unrelated: { label: 'asked: other key', color: 'var(--tok-query2)' },
};

const MIN_RUN = 3; // filler runs longer than this collapse into one gap

type Segment = { kind: 'token'; index: number } | { kind: 'gap'; count: number; from: number };

/** Segments in order: every non-filler token, with long filler runs collapsed into one gap. */
function segmentsOf(roles: string[]): Segment[] {
  const out: Segment[] = [];
  let i = 0;
  while (i < roles.length) {
    if (roles[i] === 'filler') {
      let j = i;
      while (j < roles.length && roles[j] === 'filler') j += 1;
      const run = j - i;
      if (run > MIN_RUN) out.push({ kind: 'gap', count: run, from: i });
      else for (let k = i; k < j; k += 1) out.push({ kind: 'token', index: k });
      i = j;
    } else {
      out.push({ kind: 'token', index: i });
      i += 1;
    }
  }
  return out;
}

/** One real test example, drawn as role-coloured chips. Token ids are not shown, only their roles. */
export function ExampleStrip({ example }: { example: OverwriteExample }) {
  const segments = segmentsOf(example.roles);
  const used = [...new Set(example.roles)].filter((r) => r in ROLE_INFO);
  return (
    <div className="ac-example">
      <div className="ac-example__strip" role="img" aria-label="One test example, with each token coloured by its role">
        {segments.map((seg) =>
          seg.kind === 'gap' ? (
            <span key={`g${seg.from}`} className="ac-gap" title={`${seg.count} filler tokens`}>
              {seg.count} filler
            </span>
          ) : (
            <span
              key={seg.index}
              className="ac-chip"
              style={{ background: ROLE_INFO[example.roles[seg.index]]?.color ?? 'var(--tok-filler)' }}
              title={ROLE_INFO[example.roles[seg.index]]?.label ?? ''}
            >
              {ROLE_INFO[example.roles[seg.index]]?.label ?? ''}
            </span>
          ),
        )}
      </div>
      <ul className="ac-legend" aria-label="Colour key">
        {used.map((r) => (
          <li key={r}>
            <span className="ac-swatch" style={{ background: ROLE_INFO[r].color }} />
            {ROLE_INFO[r].label}
          </li>
        ))}
      </ul>
    </div>
  );
}

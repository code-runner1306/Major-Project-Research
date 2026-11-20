import { useEffect, useState } from 'react';
import { motion, useReducedMotion } from 'motion/react';
import { type Arm, type TryRequest, type TryResponse } from '../../api';
import { Panels } from '../../components/Panels';
import { TokenStrip } from '../../components/TokenStrip';
import { postJson } from '../../components/useApi';
import { WhatThisShows } from '../../components/WhatThisShows';
import { armColor, armName, whichWriteText } from './derive';
import { PlaybackControls, StatePlayer, usePlayback } from './StatePlayer';

const URL = '/api/layout-fix/try';
const SHARED: Omit<TryRequest, 'arm'> = {
  variant: 'scalar',
  seed: 42,
  layout: 'block_first',
  rewrites: 3,
  distance: 8,
  example_seed: 7,
};
const ARMS_COMPARED: Arm[] = ['A0_control', 'A4_distance_and_rewrites'];

/** Two checkpoints (A0 and A4) run on the same block-first example, with one shared scrubber. */
export function InsideCompare() {
  const [runs, setRuns] = useState<Record<string, TryResponse> | null>(null);
  const [error, setError] = useState<string | null>(null);
  const reduce = useReducedMotion() ?? false;

  useEffect(() => {
    let cancelled = false;
    Promise.all(ARMS_COMPARED.map((arm) => postJson<TryResponse>(URL, { ...SHARED, arm } satisfies TryRequest)))
      .then(([a, b]) => {
        if (!cancelled) setRuns({ [ARMS_COMPARED[0]]: a, [ARMS_COMPARED[1]]: b });
      })
      .catch((err: unknown) => {
        if (!cancelled) setError(err instanceof Error ? err.message : String(err));
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const a = runs?.[ARMS_COMPARED[0]];
  const b = runs?.[ARMS_COMPARED[1]];
  const length = a && b ? Math.min(a.example.tokens.length, b.example.tokens.length) : 0;
  const pb = usePlayback(length);

  if (error) {
    return (
      <div className="lf-error" role="alert">
        <strong>Could not run the two checkpoints.</strong>
        <p>{error}</p>
        <p className="lf-muted">If the server is down, start it with examiner_app/start.ps1.</p>
      </div>
    );
  }
  if (!a || !b) return <div className="lf-skeleton lf-skeleton--tall" aria-label="running both checkpoints" />;

  const answer = (r: TryResponse) => r.example.queries.find((q) => q.kind === 'latest');
  const ansA = answer(a);
  const ansB = answer(b);
  const colorA = armColor(ARMS_COMPARED[0]);
  const colorB = armColor(ARMS_COMPARED[1]);

  const pair = (view: 'memory' | 'size') => (
    <motion.div
      className="lf-inside__pair"
      initial={reduce ? false : { opacity: 0 }}
      animate={{ opacity: 1 }}
      transition={{ duration: 0.4 }}
    >
      <StatePlayer response={a} label={armName(ARMS_COMPARED[0])} color={colorA} playback={pb} view={view} compact showTokens={false} />
      <StatePlayer response={b} label={armName(ARMS_COMPARED[1])} color={colorB} playback={pb} view={view} compact showTokens={false} />
    </motion.div>
  );

  return (
    <div className="lf-inside">
      <PlaybackControls pb={pb} />
      <Panels
        label="Inside views"
        items={[
          { label: 'Memory and gates', content: pair('memory') },
          { label: 'Memory size', content: pair('size') },
          {
            label: 'Tokens',
            content: (
              <div className="lf-inside__tokens">
                <TokenStrip tokens={a.example.tokens} roles={a.example.roles} selected={pb.t} onSelect={(i) => { pb.setPlaying(false); pb.setT(i); }} />
              </div>
            ),
          },
        ]}
      />
      <div className="lf-inside__answers">
        {[
          { name: armName(ARMS_COMPARED[0]), q: ansA, color: colorA },
          { name: armName(ARMS_COMPARED[1]), q: ansB, color: colorB },
        ].map((row) => (
          <p key={row.name} style={{ borderLeftColor: row.color }}>
            <strong>{row.name}</strong> at the latest query {row.q ? whichWriteText(row.q.which_write) : 'had no latest query'}.
          </p>
        ))}
      </div>
      <WhatThisShows>both models read the same example; the grids show each model's memory at the current token.</WhatThisShows>
    </div>
  );
}

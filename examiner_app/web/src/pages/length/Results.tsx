import { AnimatedBars } from '../../components/AnimatedBars';
import { AnimatedLines } from '../../components/AnimatedLines';
import { Panels } from '../../components/Panels';
import { Stat } from '../../components/Stat';
import { Remote, type RemoteState } from './common';
import { pct, type LengthSummary } from './types';

/** Results visual: three saved-result views, one at a time. Every number comes from the saved sweep. */
export function LengthResultsVisual({ state }: { state: RemoteState<LengthSummary> }) {
  return (
    <Remote state={state} what="the saved results" tall>
      {(s) => {
        const longest = s.lengths[s.lengths.length - 1];
        const groups: { init: string; variant: string }[] = [
          { init: 'default', variant: 'scalar' },
          { init: 'default', variant: 'channel-wise' },
          { init: 'mean-decay', variant: 'scalar' },
          { init: 'mean-decay', variant: 'channel-wise' },
        ];
        const series = groups.map((g) => ({
          name: `${g.init}, ${g.variant}`,
          color: g.variant === 'channel-wise' ? 'var(--c-channel)' : 'var(--c-scalar)',
          dashed: g.init === 'mean-decay',
          ys: s.lengths.map((length) => {
            const values = s.cells
              .filter((c) => c.init === g.init && c.variant === g.variant && c.length === length)
              .map((c) => c.accuracy);
            return values.reduce((a, b) => a + b, 0) / Math.max(values.length, 1);
          }),
        }));
        const bars = s.models
          .map((m) => ({
            label: `${m.init} · ${m.variant} · seed ${m.seed}`,
            value: m.accuracy[String(longest)] ?? 0,
            color: m.variant === 'channel-wise' ? 'var(--c-channel)' : 'var(--c-scalar)',
          }))
          .sort((a, b) => b.value - a.value);
        const replaysMatching = s.replays.filter((r) => r.matches_sweep === true).length;
        const cellsAgree = s.verification.verified_cells_agree;
        const hashes = s.verification.checkpoint_hashes ?? 0;

        return (
          <div className="ln-results">
            <Panels
              label="Results"
              items={[
                {
                  label: 'Average by length',
                  content: (
                    <section className="ln-panel">
                      <h3 className="ln-h3">Average accuracy by input length</h3>
                      <AnimatedLines
                        series={series}
                        xs={s.lengths}
                        yMax={1}
                        xLabel="input length (tokens)"
                        yLabel="exact accuracy"
                      />
                    </section>
                  ),
                },
                {
                  label: `Each model at ${longest.toLocaleString()}`,
                  content: (
                    <section className="ln-panel">
                      <h3 className="ln-h3">Each model at {longest.toLocaleString()} tokens</h3>
                      <div className="ln-bars2">
                        <AnimatedBars data={bars.slice(0, Math.ceil(bars.length / 2))} max={1} format={(v) => pct(v)} />
                        <AnimatedBars data={bars.slice(Math.ceil(bars.length / 2))} max={1} format={(v) => pct(v)} />
                      </div>
                    </section>
                  ),
                },
                {
                  label: 'Checks',
                  content: (
                    <section className="ln-panel">
                      <h3 className="ln-h3">Saved numbers checked</h3>
                      <div className="ln-stats">
                        <div className="ln-stat-card">
                          <Stat value={cellsAgree} format={(v) => `${Math.round(v)} / ${s.verification.cells}`} label="sweep cells agree across both saved files" />
                        </div>
                        <div className="ln-stat-card">
                          <Stat value={hashes} format={(v) => `${Math.round(v)} / ${s.verification.local_checkpoints}`} label="checkpoint hashes match" />
                        </div>
                        <div className="ln-stat-card">
                          <Stat value={replaysMatching} format={(v) => `${Math.round(v)} / ${s.replays.length}`} label="targeted replays match the sweep" />
                        </div>
                      </div>
                    </section>
                  ),
                },
              ]}
            />
          </div>
        );
      }}
    </Remote>
  );
}

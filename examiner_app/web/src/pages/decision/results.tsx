import { AnimatedBars, type BarDatum } from '../../components/AnimatedBars';
import { Panels } from '../../components/Panels';
import { WhatThisShows } from '../../components/WhatThisShows';
import { Loaded } from './visuals';
import { fixed, variantName, type DecisionSummary } from './data';

type State = Parameters<typeof Loaded>[0]['state'];

const COLOR = { scalar: 'var(--c-scalar)', 'channel-wise': 'var(--c-channel)' } as const;
const runShort = (id: string) => (id === 'short' ? 'Short' : 'Longer');
const variantShort = (v: 'scalar' | 'channel-wise') => (v === 'channel-wise' ? 'channel-wise' : 'scalar');

/** A pair of plain yes/no facts about what the saved reports contain. */
function Facts({ s }: { s: DecisionSummary }) {
  const yesNo = (b: boolean) => (b ? 'Yes' : 'No');
  return (
    <dl className="dc-facts">
      <div>
        <dt>Saved before-and-after pick comparison</dt>
        <dd>{yesNo(s.pick_comparison_recorded)}</dd>
      </div>
      <div>
        <dt>Held-out loss from before training</dt>
        <dd>{yesNo(s.pre_training_validation_recorded)}</dd>
      </div>
    </dl>
  );
}

/** Results: training loss, held-out loss and cost, one view at a time. */
export function ResultsVisual({ state }: { state: State }) {
  return (
    <Loaded state={state} what="the saved results">
      {(s) => {
        const trainData: BarDatum[] = s.runs.flatMap((run) =>
          run.variants.flatMap((v) => [
            { label: `${runShort(run.id)} ${variantShort(v.variant)}: start`, value: v.first_train_loss, color: COLOR[v.variant] },
            { label: `${runShort(run.id)} ${variantShort(v.variant)}: end`, value: v.last_train_loss, color: COLOR[v.variant] },
          ]),
        );
        const heldData: BarDatum[] = s.runs.flatMap((run) =>
          run.variants.map((v) => ({
            label: `${runShort(run.id)} ${variantShort(v.variant)}`,
            value: v.mean_validation_soft_loss,
            color: COLOR[v.variant],
          })),
        );
        const secondsData: BarDatum[] = s.runs.flatMap((run) =>
          run.variants.map((v) => ({
            label: `${runShort(run.id)} ${variantShort(v.variant)}`,
            value: v.steady_step_median_seconds,
            color: COLOR[v.variant],
          })),
        );
        const memData: BarDatum[] = s.runs.flatMap((run) =>
          run.variants.map((v) => ({
            label: `${runShort(run.id)} ${variantShort(v.variant)}`,
            value: v.peak_allocated_gib ?? 0,
            color: COLOR[v.variant],
          })),
        );
        const cases = s.runs[0]?.validation_examples ?? 0;

        return (
          <div className="dc-results">
            <Facts s={s} />
            <Panels
              label="Results"
              items={[
                {
                  label: 'Training loss',
                  content: (
                    <section className="dc-panel">
                      <h3 className="dc-h3">Training loss fell in every saved run</h3>
                      <AnimatedBars data={trainData} format={(v) => fixed(v, 2)} />
                      <WhatThisShows>training loss at the first and last step, for each run and decay granularity.</WhatThisShows>
                    </section>
                  ),
                },
                {
                  label: 'Held-out loss',
                  content: (
                    <section className="dc-panel">
                      <h3 className="dc-h3">Loss on held-out cases after training</h3>
                      <AnimatedBars data={heldData} format={(v) => fixed(v, 3)} />
                      <WhatThisShows>
                        Soft loss on {cases} held-out cases. The saved files have no value from before training.
                      </WhatThisShows>
                    </section>
                  ),
                },
                {
                  label: 'Cost',
                  content: (
                    <section className="dc-panel">
                      <h3 className="dc-h3">Time and memory per training step on the T4 GPU</h3>
                      <div className="dc-two">
                        <div>
                          <span className="dc-kicker">Seconds per step</span>
                          <AnimatedBars data={secondsData} format={(v) => `${fixed(v, 1)} s`} />
                        </div>
                        <div>
                          <span className="dc-kicker">Peak GPU memory</span>
                          <AnimatedBars data={memData} format={(v) => `${fixed(v, 2)} GiB`} />
                        </div>
                      </div>
                      <WhatThisShows>measured on one T4 GPU; the two runs used different batch sizes and lengths.</WhatThisShows>
                    </section>
                  ),
                },
              ]}
            />
          </div>
        );
      }}
    </Loaded>
  );
}

/** Try it slot: every saved number in one browsable table. */
export function SavedTable({ state }: { state: State }) {
  return (
    <Loaded state={state} what="the saved numbers">
      {(s) => (
        <div className="dc-table-wrap">
          <table className="dc-table">
            <thead>
              <tr>
                <th scope="col">Run</th>
                <th scope="col">Decay granularity</th>
                <th scope="col">Start loss</th>
                <th scope="col">End loss</th>
                <th scope="col">Held-out loss</th>
                <th scope="col">Seconds per step</th>
                <th scope="col">Peak GPU GiB</th>
              </tr>
            </thead>
            <tbody>
              {s.runs.flatMap((run) =>
                run.variants.map((v) => (
                  <tr key={`${run.id}-${v.variant}`}>
                    <td>
                      {run.steps} steps, batch {run.batch_size}, length {run.max_length}
                    </td>
                    <td>{variantName(v.variant)}</td>
                    <td>{fixed(v.first_train_loss)}</td>
                    <td>{fixed(v.last_train_loss)}</td>
                    <td>{fixed(v.mean_validation_soft_loss)}</td>
                    <td>{fixed(v.steady_step_median_seconds, 2)}</td>
                    <td>{v.peak_allocated_gib === null ? '—' : fixed(v.peak_allocated_gib, 2)}</td>
                  </tr>
                )),
              )}
            </tbody>
          </table>
          <WhatThisShows>every number from the two saved reports, unchanged.</WhatThisShows>
        </div>
      )}
    </Loaded>
  );
}

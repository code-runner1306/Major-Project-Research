import type { ReactNode } from 'react';
import { ExampleError, Skeleton, WaitingForServer } from '../dataset/shared';
import { Unavailable } from '../../components/Unavailable';
import { plainLabel, type DecisionSummary } from './data';
import './decision.css';

type State = { data: DecisionSummary | null; error: { message: string; status: number | null } | null; waiting: boolean; pending: boolean };

/** Loading, retry notice, error panel, then the children with the loaded summary. */
export function Loaded({ state, what, children }: { state: State; what: string; children: (s: DecisionSummary) => ReactNode }) {
  if (state.error) return <ExampleError error={state.error} what={`${what} could not be loaded.`} />;
  if (state.waiting) return <WaitingForServer />;
  if (!state.data) return <Skeleton label={`loading ${what}`} />;
  return <>{children(state.data)}</>;
}

/** Slide 1: a customer case goes in, one of five actions comes out. */
export function QuestionVisual({ state }: { state: State }) {
  return (
    <Loaded state={state} what="the question">
      {(s) => (
        <div className="dc-flow dc-question">
          <div className="dc-card dc-card--case">
            <span className="dc-kicker">Input</span>
            <strong>One customer-service case</strong>
            <span className="dc-muted">Written as text: what happened and what the record shows.</span>
          </div>
          <div className="dc-arrow" aria-hidden="true" />
          <div className="dc-card dc-card--options">
            <span className="dc-kicker">The model picks one of {s.question.labels.length}</span>
            <ol className="dc-options">
              {s.question.labels.map((label) => (
                <li key={label}>{plainLabel(label)}</li>
              ))}
            </ol>
          </div>
        </div>
      )}
    </Loaded>
  );
}

/** Slide 2: the same small model, two decay granularities, each with a fresh five-way head. */
export function ModelVisual({ state }: { state: State }) {
  return (
    <Loaded state={state} what="the model setup">
      {(s) => (
        <div className="dc-model">
          {(['scalar decay', 'channel-wise decay'] as const).map((name) => (
            <div className="dc-lane" key={name}>
              <span className="dc-lane__name">{name}</span>
              <div className="dc-steps">
                <div className="dc-step">
                  <span className="dc-kicker">Start</span>
                  <strong>Trained 30M model</strong>
                </div>
                <div className="dc-arrow dc-arrow--small" aria-hidden="true" />
                <div className="dc-step">
                  <span className="dc-kicker">Extra round</span>
                  <strong>{s.runs.map((r) => `${r.steps} steps`).join(' or ')}</strong>
                </div>
                <div className="dc-arrow dc-arrow--small" aria-hidden="true" />
                <div className="dc-step dc-step--head">
                  <span className="dc-kicker">New head</span>
                  <strong>{s.question.labels.length} scores</strong>
                </div>
              </div>
            </div>
          ))}
        </div>
      )}
    </Loaded>
  );
}

/** Slide 5: the prompt as the feasibility script builds it, schematic only (no example data is saved). */
export function InsideVisual({ state }: { state: State }) {
  return (
    <Loaded state={state} what="the prompt layout">
      {(s) => (
        <div className="dc-inside">
          <pre className="dc-prompt" aria-label="Prompt layout">
            <span className="dc-prompt__key">State:</span> {'<customer case, cut at the end if long>'}
            {'\n'}
            <span className="dc-prompt__key">Question:</span> {'<what to decide>'}
            {'\n'}
            <span className="dc-prompt__key">Options:</span>
            {s.question.labels.map((label) => (
              <span key={label} className="dc-prompt__line">
                {'\n'}
                {label}: {'<description>'}
              </span>
            ))}
            {'\n'}
            <span className="dc-prompt__key">Decision:</span>
          </pre>
          <div className="dc-inside__chain">
            <span>Text to tokens</span>
            <span className="dc-arrow dc-arrow--small" aria-hidden="true" />
            <span>30M model, last token</span>
            <span className="dc-arrow dc-arrow--small" aria-hidden="true" />
            <span>New head</span>
            <span className="dc-arrow dc-arrow--small" aria-hidden="true" />
            <span>{s.question.labels.length} scores, one per option</span>
          </div>
        </div>
      )}
    </Loaded>
  );
}

/** Slide 6: no fine-tuned weights were saved, so there is nothing to run. */
export function TryUnavailable({ state }: { state: State }) {
  return (
    <Loaded state={state} what="the saved numbers">
      {(s) => (
        <Unavailable
          reason={
            s.fine_tuned_weights_saved
              ? 'Fine-tuned weights were saved, but they are not loaded on this page.'
              : 'Models unavailable on this machine: the extra round never saved fine-tuned weights.'
          }
        />
      )}
    </Loaded>
  );
}

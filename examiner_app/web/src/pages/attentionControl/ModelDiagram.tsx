import type { AttentionControlSummary } from './types';

const count = (v: number) => v.toLocaleString('en-GB');

/** Two memory designs on one shared backbone. Numbers come from the saved summary (backbone settings). */
export function ModelDiagram({ summary }: { summary: AttentionControlSummary }) {
  const m = summary.memory;
  const runs = summary.same_protocol.overwrite;
  return (
    <div className="ac-model">
      <div className="ac-backbone" aria-label="Shared backbone">
        {m.layers} layers · width {m.width} · same feed-forward blocks, norms and embeddings
      </div>
      <div className="ac-model__cards">
        <figure className="ac-card">
          <figcaption>Recurrent memory</figcaption>
          <div className="ac-card__art" aria-hidden="true">
            <div className="ac-table">
              {Array.from({ length: m.heads }, (_, i) => (
                <span key={i} className="ac-table__cell">
                  {m.head_width}×{m.head_width}
                </span>
              ))}
            </div>
            <span className="ac-card__note">one fixed-size table, updated token by token</span>
          </div>
          <p className="ac-card__stat">
            {count(m.recurrent_state_numbers_per_layer)} numbers per layer, at any length
          </p>
        </figure>
        <figure className="ac-card">
          <figcaption>Attention</figcaption>
          <div className="ac-card__art" aria-hidden="true">
            <div className="ac-tokens">
              {Array.from({ length: 8 }, (_, i) => (
                <span key={i} className="ac-tokens__cell" />
              ))}
            </div>
            <span className="ac-card__note">every earlier token is kept and can be read</span>
          </div>
          <p className="ac-card__stat">
            +{count(m.attention_cache_numbers_per_token_per_layer)} numbers per token, per layer
          </p>
        </figure>
      </div>
      <p className="ac-model__params">
        Parameters: attention {count(runs.attention.parameters)}, recurrent {count(runs.gdn.parameters)}
      </p>
    </div>
  );
}

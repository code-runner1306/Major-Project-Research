import type { ParametersResponse, Variant } from './types';
import { fmtInt } from './common';

/** One decay gate drawn as cells: one cell per gate value per token (1 for scalar, 128 for channel-wise). */
function GateCells({ width }: { width: number }) {
  const columns = width === 1 ? 1 : 16;
  return (
    <div className="cost-cells" style={{ gridTemplateColumns: `repeat(${columns}, 10px)` }} aria-hidden="true">
      {Array.from({ length: width }, (_, i) => (
        <span key={i} className="cost-cell" />
      ))}
    </div>
  );
}

function ArchColumn({ variant, p }: { variant: Variant; p: ParametersResponse }) {
  const arch = p.architectures[variant];
  const layer = arch.layers[0];
  const label = variant === 'scalar' ? 'Scalar decay' : 'Channel-wise decay';
  return (
    <div className={`cost-arch cost-arch--${variant === 'scalar' ? 'scalar' : 'channel'}`}>
      <h4>{label}</h4>
      <p className="cost-arch-count">
        {layer.gate_width === 1 ? '1 gate value' : `${layer.gate_width} gate values`} <span>per token, per layer</span>
      </p>
      <GateCells width={layer.gate_width} />
      <p className="cost-arch-weights">
        Gate weights per layer: <strong>{fmtInt(layer.a_proj_parameters)}</strong>
      </p>
      <p className="cost-arch-weights">
        Model total: <strong>{fmtInt(arch.total_parameters)}</strong>
      </p>
    </div>
  );
}

/** Scalar and channel-wise models side by side, with the per-layer arithmetic and the share of the model that is extra. */
export function GateVisual({ p }: { p: ParametersResponse }) {
  const sc = p.architectures.scalar;
  const ch = p.architectures['channel-wise'];
  const s0 = sc.layers[0];
  const c0 = ch.layers[0];
  const dA = c0.a_proj_parameters - s0.a_proj_parameters;
  const dLog = c0.A_log_parameters - s0.A_log_parameters;
  const dDt = c0.dt_bias_parameters - s0.dt_bias_parameters;
  const layers = sc.layers.length;
  const widthRatio = (sc.total_parameters / ch.total_parameters) * 100;
  const extraRatio = 100 - widthRatio;

  return (
    <div className="cost-gate">
      <div className="cost-arch-row">
        <ArchColumn variant="scalar" p={p} />
        <ArchColumn variant="channel-wise" p={p} />
      </div>

      <div className="cost-share" role="img" aria-label={`Channel-wise model has ${fmtInt(p.delta.parameters)} more parameters`}>
        <div className="cost-share-row">
          <span className="cost-share-name">Scalar</span>
          <div className="cost-share-track">
            <div className="cost-share-base" style={{ width: `${widthRatio}%` }} />
          </div>
          <span className="cost-share-value">{fmtInt(sc.total_parameters)}</span>
        </div>
        <div className="cost-share-row">
          <span className="cost-share-name">Channel-wise</span>
          <div className="cost-share-track">
            <div className="cost-share-base" style={{ width: `${widthRatio}%` }} />
            <div className="cost-share-extra" style={{ width: `${extraRatio}%`, left: `${widthRatio}%` }} />
          </div>
          <span className="cost-share-value">{fmtInt(ch.total_parameters)}</span>
        </div>
      </div>

      <div className="cost-sum" aria-label="Per-layer arithmetic">
        <span>Per layer: gate projection <strong>+{fmtInt(dA)}</strong></span>
        <span>decay offsets <strong>+{fmtInt(dLog)}</strong> (A_log) <strong>+{fmtInt(dDt)}</strong> (dt_bias)</span>
        <span className="cost-sum-total">
          {fmtInt(p.delta.per_layer_parameters)} × {layers} layers = <strong>+{fmtInt(p.delta.parameters)}</strong>
        </span>
      </div>

      <div className="cost-state">
        <span>Recurrent state, both models</span>
        <strong>{fmtInt(sc.state_bytes_fp32)} bytes</strong>
        <span className="cost-state-note">fp32, {layers} layers. The same size in both: the gate changes its length, not the state.</span>
      </div>
    </div>
  );
}

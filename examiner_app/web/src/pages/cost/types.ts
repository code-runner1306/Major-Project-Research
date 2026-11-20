// Shapes of /api/cost/summary (saved artifacts, see server/build_cost_export.py) and /api/cost/parameters (live CPU build).

export type Hardware = 'a6000' | 't4';
export type Family = 'mqar' | 'overwrite';
export type Mode = 'parallel' | 'sequential' | 'not recorded';
export type Variant = 'scalar' | 'channel-wise';

export interface StepRun {
  hardware: Hardware;
  family: Family;
  mode: Mode;
  variant: Variant;
  gpu: string;
  parameters: number;
  accepted_optimizer_steps: number;
  effective_batch: number;
  length: number;
  precision: string;
  microbatch: number;
  mean_step_seconds: number;
  median_step_seconds: number;
  min_step_seconds: number;
  max_step_seconds: number;
  sd_step_seconds: number;
  se_step_seconds: number;
  examples_per_second: number;
  setup_seconds: number;
  validation_seconds: number;
  checkpoint_seconds: number;
  wall_seconds: number;
  step_seconds: number[];
  source: string;
}

export interface ProjectionJob {
  family: Family;
  mode: 'parallel' | 'sequential';
  variant: Variant;
  optimizer_steps: number;
  estimated_seconds: number;
  conservative_seconds: number;
  mean_step_seconds: number;
}

export interface ProjectionPair {
  family: Family;
  mode: 'parallel' | 'sequential';
  estimated_pair_seconds: number;
  conservative_pair_seconds: number;
}

export interface FullResearchRun {
  family: Family;
  variant: Variant;
  seed: number;
  parameters: number;
  gate_parameter_overhead: number;
  device: string | null;
  training_wall_seconds: number;
  training_steps: number;
  training_tokens_per_second: number;
  mean_step_seconds: number;
  peak_allocated_bytes: number | null;
  energy_supported: boolean;
  energy_joules: number | null;
  energy_reason: string | null;
  decode_ms_per_token_context_256: number;
  decode_state_bytes: number;
  source: string;
}

export interface TinyStoriesRun {
  variant: Variant;
  seed: number;
  parameters: number;
  gate_parameter_overhead: number;
  device: string | null;
  training_wall_seconds: number;
  training_steps: number;
  processed_tokens: number;
  energy_supported: boolean;
  energy_joules: number | null;
  energy_method: string | null;
  energy_reason: string | null;
  source: string;
}

export interface CostSummary {
  meta: { generator: string; batch_examples_per_step: number; recipe_model: string; hardware_note: string };
  hardware: { id: Hardware; label: string; kind: string; source: string }[];
  excluded: { source: string; reason: string }[];
  step_timing: StepRun[];
  projection: {
    source: string;
    optimizer_steps: number;
    epochs: number;
    jobs: ProjectionJob[];
    pairs: ProjectionPair[];
    recommended_mode: Record<Family, string>;
  };
  joint_measured: {
    source: string;
    hardware: Hardware;
    optimizer_steps_per_model: number;
    epochs: number;
    paired_sessions: number;
    measured_pair_seconds: Record<Family, number>;
    note: string;
  };
  other_models: {
    note: string;
    full_research_10m: FullResearchRun[];
    tinystories_10m: TinyStoriesRun[];
  };
  not_recorded: { recipe_energy: string; recipe_memory: string };
  provenance: { file: string; generator: string };
}

export interface GateLayer {
  layer: number;
  gate_width: number;
  a_proj_parameters: number;
  a_proj_shape: number[];
  A_log_parameters: number;
  dt_bias_parameters: number;
  gate_parameters: number;
  gate_macs_per_token: number;
  state_shape: number[];
  state_bytes_fp32: number;
}

export interface ArchFacts {
  architecture: Variant;
  total_parameters: number;
  gate_parameters: number;
  layers: GateLayer[];
  gate_macs_per_token: number;
  linear_macs_per_token: number;
  state_bytes_fp32: number;
}

export interface ParametersResponse {
  architectures: { scalar: ArchFacts; 'channel-wise': ArchFacts };
  delta: {
    parameters: number;
    percent_of_scalar: number;
    per_layer_parameters: number;
    extra_gate_macs_per_token: number;
    extra_gate_macs_share_of_linear: number;
    state_bytes_difference: number;
  };
  settings: { vocabulary: number; length: number; width: number; backend: string; seed: number; layers: number; heads: number };
  source: string;
}

// Response shapes for /api/attention-control/* (see content_briefs/attention-control.md for sources).
// Accuracies are fractions (0 to 1).

export const SUMMARY_URL = '/api/attention-control/summary';
export const EXAMPLE_URL = '/api/attention-control/example';

export interface Counts {
  correct: number;
  total: number;
  accuracy: number;
}

export interface SameRun {
  answer_free: Counts;
  answer_free_detail: {
    first_query_accuracy: number;
    later_query_accuracy: number;
    latest_value_accuracy: number | null;
    unrelated_association_retention: number | null;
    stale_value_error_rate: number | null;
  };
  development_curve: { step: number; accuracy: number }[];
  parameters: number;
  clipped_step_fraction: number;
  steps: number;
  wall_seconds: number;
  status: string;
  key_free_baselines: { name: string; accuracy: number; correct: number; total: number };
}

export interface KaggleRun {
  answer_free: Counts;
  final_validation_loss: number;
  parameters: number;
}

export interface AttentionControlSummary {
  status: { live: boolean; checkpoints: string[]; reason: string };
  protocol: {
    same: {
      steps: number;
      learning_rate: number;
      training_examples: number;
      held_out_examples: number;
      sequence_length: number;
      keys: number;
      query_distance: number;
      overwrite_count: number;
      vocabulary: number;
    };
    kaggle: { steps: number; learning_rate: number; held_out_examples: number; context_length: number; seed: number };
  };
  same_protocol: Record<'mqar' | 'overwrite', Record<'gdn' | 'attention', SameRun>>;
  scalar_vs_channel: Record<'mqar' | 'overwrite', Record<'C1' | 'C3', KaggleRun>>;
  paper_context: { channel_minus_scalar_gap: { '10M': number | null; '100M': number | null }; note: string };
  memory: {
    recurrent_state_numbers_per_layer: number;
    attention_cache_numbers_per_token_per_layer: number;
    heads: number;
    head_width: number;
    layers: number;
    width: number;
  };
  headline: { value: string; label: string };
  provenance: { files: string[] };
}

export interface ExampleQuery {
  kind: 'latest' | 'unrelated';
  query_position: number;
  key_position: number;
  answer: number;
  latest_value_position: number | null;
}

export interface OverwriteExample {
  index: number;
  seed: number;
  sequence_length: number;
  token_count: number;
  tokens: number[];
  roles: string[];
  queries: ExampleQuery[];
  writes: { target: number[]; all: number[]; key_positions: number[] };
}

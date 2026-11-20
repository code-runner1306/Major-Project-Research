// Response shapes for GET /api/datasets and POST /api/datasets/generate (see ../../../../SPEC.md, "HTTP API").

export type Origin = 'port' | 'ours' | 'upstream' | 'external';

export type ParamValue = number | string;

export interface ParamSpec {
  type: 'int' | 'choice';
  min?: number;
  max?: number;
  choices?: string[];
  default: ParamValue;
}

export interface GeneratorInfo {
  id: string;
  name: string;
  source: string;
  origin: Origin;
  live: boolean;
  params: Record<string, ParamSpec>;
}

export interface ScoredPosition {
  position: number;
  answer: number;
  kind: string;
}

export interface DatasetExample {
  generator: string;
  tokens: number[];
  roles: string[];
  scored: ScoredPosition[];
  /** Plain-English reading, one sentence per fact. */
  notes: string[];
  meta: Record<string, unknown>;
  /** Token id (as a string key) to what it means in this generator. */
  reserved_tokens: Record<string, string>;
  /** Decoded word piece per token (TinyStories and other text generators). Same length as tokens. */
  token_text?: string[];
  /** The story text (first 800 characters), for text generators. */
  text?: string;
}

export interface GenerateRequest {
  generator: string;
  seed: number;
  params: Record<string, ParamValue>;
}

export const DATASETS_URL = '/api/datasets';
export const GENERATE_URL = '/api/datasets/generate';
export const SERVER_HINT = 'Start the server: examiner_app/start.ps1';

export const ORIGIN_LABEL: Record<Origin, string> = {
  port: 'ported',
  ours: 'ours',
  upstream: 'upstream',
  external: 'external',
};

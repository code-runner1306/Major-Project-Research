import { useEffect, useRef, useState } from 'react';
import { TokenStrip } from '../../components/TokenStrip';
import { Term } from '../../components/Term';
import {
  DATASETS_URL,
  GENERATE_URL,
  ORIGIN_LABEL,
  SERVER_HINT,
  type DatasetExample,
  type GenerateRequest,
  type GeneratorInfo,
  type Origin,
  type ParamValue,
} from './types';
import './dataset.css';

/** A failed request. status is null when the server could not be reached at all. */
export class ApiFailure extends Error {
  status: number | null;

  constructor(message: string, status: number | null) {
    super(message);
    this.name = 'ApiFailure';
    this.status = status;
  }
}

/** Plain-English names for every role, used in legends and readings. */
export const ROLE_NAME: Record<string, string> = {
  filler: 'filler',
  key: 'key',
  value: 'value',
  target_key: 'key of the queried value',
  target_value: 'value of the queried key',
  marker: 'query marker',
  query_latest: 'query for the latest value',
  query_unrelated: 'query for an unrelated key',
  query_key: 'query key',
  answer_slot: 'answer slot',
  noise: 'random noise',
  decoy_key: 'decoy key',
  decoy_value: 'decoy value',
  mask: 'masked (to be filled in)',
  text: 'story text',
  eos: 'end of text',
};

const KEY_ROLES = new Set(['key', 'target_key']);
const VALUE_ROLES = new Set(['value', 'target_value']);
export const isKeyRole = (role: string) => KEY_ROLES.has(role);
export const isValueRole = (role: string) => VALUE_ROLES.has(role);

export const RETRY_INTERVAL_MS = 2000;
export const RETRY_LIMIT = 10;

/** The API is not up yet: nothing answered at all (status null), or a proxy in front of it returned 502. */
export function isServerStarting(error: ApiFailure | null): boolean {
  return error !== null && (error.status === null || error.status === 502);
}

interface RetryState<T> {
  key: string;
  attempt: number;
  data: T | null;
  error: ApiFailure | null;
}

/**
 * Runs `run` for `key` after `delay` ms. While the server is starting (no answer, or 502), a failed call is
 * retried every 2 s, up to 10 times, and `waiting` is true meanwhile. Only then is the failure reported as an error.
 * `data` keeps the last result while a new key loads (`pending` is true).
 */
export function useRetrying<T>(key: string | null, run: () => Promise<T>, delay = 0) {
  const [state, setState] = useState<RetryState<T>>({ key: '', attempt: 0, data: null, error: null });
  const runRef = useRef(run);
  useEffect(() => {
    runRef.current = run;
  });

  const attempt = state.key === key ? state.attempt : 0;

  useEffect(() => {
    if (key === null) return;
    let cancelled = false;
    let retryTimer: number | undefined;
    const timer = window.setTimeout(
      () => {
        runRef.current().then(
          (data) => {
            if (!cancelled) setState({ key, attempt, data, error: null });
          },
          (err: unknown) => {
            if (cancelled) return;
            const failure = err instanceof ApiFailure ? err : new ApiFailure(String(err), 0);
            setState({ key, attempt, data: null, error: failure });
            if (isServerStarting(failure) && attempt < RETRY_LIMIT) {
              retryTimer = window.setTimeout(() => {
                if (!cancelled) setState({ key, attempt: attempt + 1, data: null, error: failure });
              }, RETRY_INTERVAL_MS);
            }
          },
        );
      },
      attempt === 0 ? delay : 0,
    );
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
      window.clearTimeout(retryTimer);
    };
  }, [key, attempt, delay]);

  if (key === null) return { data: null, error: null, waiting: false, pending: false };
  const pending = state.key !== key;
  const waiting = !pending && isServerStarting(state.error) && state.attempt < RETRY_LIMIT;
  return {
    data: state.error ? null : state.data,
    error: pending || waiting ? null : state.error,
    waiting,
    pending,
  };
}

/** Shown while a request waits for the model server to start. */
export function WaitingForServer() {
  return (
    <p className="ds-waiting" role="status">
      Waiting for the model server to start…
    </p>
  );
}

/** GET a dataset JSON resource. A network failure becomes an ApiFailure with status null. */
async function getJson<T>(url: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url);
  } catch {
    throw new ApiFailure('The server did not answer.', null);
  }
  if (!res.ok) throw new ApiFailure(await readDetail(res), res.status);
  return (await res.json()) as T;
}

export function useGenerators() {
  return useRetrying<GeneratorInfo[]>(DATASETS_URL, () => getJson<GeneratorInfo[]>(DATASETS_URL));
}

export function defaultParams(info: GeneratorInfo): Record<string, ParamValue> {
  return Object.fromEntries(Object.entries(info.params).map(([name, spec]) => [name, spec.default]));
}

/** Reads the server's error detail: a string, or a list of validation messages (422). */
async function readDetail(res: Response): Promise<string> {
  try {
    const body = (await res.json()) as { detail?: unknown };
    const detail = body.detail;
    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item: unknown) => {
          if (typeof item === 'object' && item !== null && 'msg' in item) {
            const loc = 'loc' in item && Array.isArray(item.loc) ? item.loc.filter((p) => p !== 'body').join('.') : '';
            return loc ? `${loc}: ${String(item.msg)}` : String(item.msg);
          }
          return JSON.stringify(item);
        })
        .join('; ');
    }
    if (detail !== undefined) return JSON.stringify(detail);
  } catch {
    // Non-JSON error body: fall back to the status line.
  }
  return `${res.status} ${res.statusText}`.trim();
}

export async function generateExample(req: GenerateRequest): Promise<DatasetExample> {
  let res: Response;
  try {
    res = await fetch(GENERATE_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(req),
    });
  } catch {
    throw new ApiFailure('The server did not answer.', null);
  }
  if (!res.ok) throw new ApiFailure(await readDetail(res), res.status);
  return (await res.json()) as DatasetExample;
}

/**
 * Generates one example for a request. Debounced: only the last settled request is sent.
 * `data` keeps the previous example while a new one loads (`pending` is true), so the UI can dim it.
 */
export function useExample(req: GenerateRequest | null, delay = 200) {
  const key = req === null ? null : JSON.stringify(req);
  const example = useRetrying<DatasetExample>(key, () => generateExample(JSON.parse(String(key)) as GenerateRequest), delay);
  return {
    idle: key === null,
    pending: example.pending,
    waiting: example.waiting,
    data: example.data,
    error: example.error,
  };
}

/** One example of a named generator with its default parameters, merged with any overrides. */
export function useLiveExample(generator: string, seed: number, overrides: Record<string, ParamValue> = {}) {
  const gens = useGenerators();
  const info = gens.data?.find((g) => g.id === generator) ?? null;
  const req = info ? { generator, seed, params: { ...defaultParams(info), ...overrides } } : null;
  const ex = useExample(req);
  return {
    ...ex,
    info,
    error: ex.error ?? gens.error,
    waiting: ex.waiting || gens.waiting,
  };
}

export function OriginBadge({ origin }: { origin: Origin }) {
  return <span className={`ds-badge ds-badge--${origin}`}>{ORIGIN_LABEL[origin]}</span>;
}

/** Error panel. A null status means the server did not answer: the message then says how to start it. */
export function ExampleError({
  error,
  what = 'The example could not be built.',
}: {
  error: { message: string; status: number | null };
  what?: string;
}) {
  if (error.status === null) {
    return (
      <div className="ds-error" role="alert">
        <strong>The server is not running.</strong>
        <p>{error.message}</p>
        <p className="ds-muted">{SERVER_HINT}</p>
      </div>
    );
  }
  const heading =
    error.status === 409
      ? 'Not available on this machine.'
      : error.status === 422
        ? 'The server rejected these settings.'
        : what;
  return (
    <div className="ds-error" role="alert">
      <strong>{heading}</strong>
      <p>{error.message}</p>
    </div>
  );
}

export function Skeleton({ label, tall = false }: { label: string; tall?: boolean }) {
  return <div className={tall ? 'ds-skeleton ds-skeleton--tall' : 'ds-skeleton'} role="status" aria-label={label} />;
}

/** Colour key for hand-built token rows (TokenStrip draws its own legend). */
export function RoleLegend({ roles }: { roles: string[] }) {
  const present = Array.from(new Set(roles));
  return (
    <div className="viz-tokens__legend" aria-label="role legend">
      {present.map((role) => (
        <span className="viz-tokens__legend-item" key={role}>
          <span className={`viz-tokens__swatch viz-token--${role}`} />
          {ROLE_NAME[role] ?? role}
        </span>
      ))}
    </div>
  );
}

export function SeedControl({ seed, onChange }: { seed: number; onChange: (seed: number) => void }) {
  return (
    <div className="ds-seed">
      <label className="ds-field ds-field--inline">
        <span className="ds-label">
          <Term term="seed" /> (the same seed gives the same example)
        </span>
        <input
          type="number"
          min={0}
          value={seed}
          onChange={(e) => onChange(Math.max(0, Math.floor(Number(e.target.value) || 0)))}
        />
      </label>
      <button type="button" className="ds-btn" onClick={() => onChange(Math.floor(Math.random() * 10000))}>
        Roll the dice
      </button>
    </div>
  );
}

/** Small preview strip with scored answers outlined. Used wherever a generator's example is shown whole. */
export function ScoredStrip({
  ex,
  compact = true,
  pending = false,
}: {
  ex: DatasetExample;
  compact?: boolean;
  pending?: boolean;
}) {
  return (
    <div className={pending ? 'ds-pending' : undefined}>
      {ex.text && (
        <figure className="ds-story">
          <figcaption>The story</figcaption>
          <p>{ex.text}</p>
        </figure>
      )}
      <TokenStrip
        tokens={ex.tokens}
        roles={ex.roles}
        scored={ex.scored.map((s) => s.position)}
        compactFiller={compact}
        tokenText={ex.token_text}
      />
    </div>
  );
}

const NUMBER_WORDS = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten'];
export const numberWord = (n: number) => (n >= 0 && n < NUMBER_WORDS.length ? NUMBER_WORDS[n] : String(n));

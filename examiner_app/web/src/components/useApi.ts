import { useEffect, useState } from 'react';

export interface ApiResult<T> {
  data: T | null;
  error: string | null;
  loading: boolean;
}

interface Settled<T> {
  url: string;
  data: T | null;
  error: string | null;
}

async function failureMessage(res: Response): Promise<string> {
  const fallback = `${res.status} ${res.statusText}`.trim();
  try {
    const body = (await res.json()) as { detail?: unknown };
    if (typeof body.detail === 'string') return body.detail;
    if (body.detail !== undefined) return JSON.stringify(body.detail);
  } catch {
    // Non-JSON error body: fall back to the status line.
  }
  return fallback;
}

/** GET a JSON resource. Pass null to skip fetching. Stale responses for a previous url are never returned. */
export function useApi<T>(url: string | null): ApiResult<T> {
  const [settled, setSettled] = useState<Settled<T> | null>(null);

  useEffect(() => {
    if (url === null) return;
    const controller = new AbortController();
    fetch(url, { signal: controller.signal })
      .then(async (res) => {
        if (!res.ok) throw new Error(await failureMessage(res));
        return (await res.json()) as T;
      })
      .then((data) => setSettled({ url, data, error: null }))
      .catch((err: unknown) => {
        if (controller.signal.aborted) return;
        setSettled({ url, data: null, error: err instanceof Error ? err.message : String(err) });
      });
    return () => controller.abort();
  }, [url]);

  if (url === null) return { data: null, error: null, loading: false };
  if (settled === null || settled.url !== url) return { data: null, error: null, loading: true };
  return { data: settled.data, error: settled.error, loading: false };
}

/** POST a JSON body and parse the JSON response; rejects with the server's detail message on failure. */
export async function postJson<T>(url: string, body: unknown): Promise<T> {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await failureMessage(res));
  return (await res.json()) as T;
}

import { useMemo, useSyncExternalStore } from 'react';

export const DEFAULT_ROUTE = 'intro';

// Fired after setRouteSection changes the hash silently (replaceState does not fire hashchange).
const ROUTE_EVENT = 'examiner:route';

export interface Route {
  /** Experiment id, e.g. "shortcut". */
  id: string;
  /** Slide id inside the experiment, e.g. "results"; null means the first slide. */
  section: string | null;
}

// Hash routes look like #/<experimentId> or #/<experimentId>/<section>. The query string (e.g. ?x=1) sits before the hash and is kept.
function parseRoute(hash: string): Route {
  const match = hash.match(/^#\/([^?#/]+)(?:\/([^?#/]+))?/);
  if (!match) return { id: DEFAULT_ROUTE, section: null };
  return {
    id: decodeURIComponent(match[1]),
    section: match[2] ? decodeURIComponent(match[2]) : null,
  };
}

function subscribe(onChange: () => void): () => void {
  window.addEventListener('hashchange', onChange);
  window.addEventListener(ROUTE_EVENT, onChange);
  return () => {
    window.removeEventListener('hashchange', onChange);
    window.removeEventListener(ROUTE_EVENT, onChange);
  };
}

function readHash(): string {
  return window.location.hash;
}

/** Current experiment id and slide from the URL hash, re-rendering on change. */
export function useHashRoute(): Route {
  const hash = useSyncExternalStore(subscribe, readHash, () => '');
  return useMemo(() => parseRoute(hash), [hash]);
}

/** Moves to a slide by rewriting the hash in place (no new history entry), then tells the router. */
export function setRouteSection(id: string, section: string): void {
  const url = `${window.location.pathname}${window.location.search}#/${encodeURIComponent(id)}/${encodeURIComponent(section)}`;
  window.history.replaceState(window.history.state, '', url);
  window.dispatchEvent(new Event(ROUTE_EVENT));
}

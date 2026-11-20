import { useEffect, useState } from 'react';

// Dev-only floating bar for previewing prototype variants. It reads and writes one URL search
// param (e.g. ?theme=paper) with history.replaceState and dispatches a 'prototype-param' window
// event on every change. Hidden in production builds.
export const PROTOTYPE_EVENT = 'prototype-param';

export interface PrototypeOption {
  key: string;
  label: string;
}

export interface PrototypeParamDetail {
  param: string;
  value: string;
}

/** Current value of a prototype param from the real query string (before the hash). */
export function getPrototypeParam(param: string, fallback: string): string {
  return new URLSearchParams(window.location.search).get(param) ?? fallback;
}

function isTyping(target: EventTarget | null): boolean {
  if (!(target instanceof HTMLElement)) return false;
  return target.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(target.tagName);
}

export function PrototypeSwitcher({
  param,
  options,
}: {
  param: string;
  options: PrototypeOption[];
}) {
  const [value, setValue] = useState(() => getPrototypeParam(param, options[0].key));
  const index = Math.max(0, options.findIndex((o) => o.key === value));
  const current = options[index];

  // Keep several instances (or a popstate) in step with the URL.
  useEffect(() => {
    const sync = () => setValue(getPrototypeParam(param, options[0].key));
    window.addEventListener('popstate', sync);
    window.addEventListener(PROTOTYPE_EVENT, sync);
    return () => {
      window.removeEventListener('popstate', sync);
      window.removeEventListener(PROTOTYPE_EVENT, sync);
    };
  }, [param, options]);

  const choose = (next: number) => {
    const key = options[(next + options.length) % options.length].key;
    const url = new URL(window.location.href);
    url.searchParams.set(param, key);
    window.history.replaceState(window.history.state, '', url.toString());
    setValue(key);
    window.dispatchEvent(
      new CustomEvent<PrototypeParamDetail>(PROTOTYPE_EVENT, { detail: { param, value: key } }),
    );
  };

  useEffect(() => {
    if (import.meta.env.PROD) return;
    const onKey = (e: KeyboardEvent) => {
      if (isTyping(e.target) || e.metaKey || e.ctrlKey || e.altKey) return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      choose(index + (e.key === 'ArrowRight' ? 1 : -1));
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  });

  if (import.meta.env.PROD) return null;

  return (
    <div className="prototype-switcher" role="toolbar" aria-label={`${param} prototype (dev only)`}>
      <button type="button" onClick={() => choose(index - 1)} aria-label="Previous option">
        ←
      </button>
      <span className="prototype-label">{current.label}</span>
      <button type="button" onClick={() => choose(index + 1)} aria-label="Next option">
        →
      </button>
    </div>
  );
}

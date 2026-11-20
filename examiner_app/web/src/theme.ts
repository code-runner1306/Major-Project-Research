import { useEffect, useMemo, useState } from 'react';

// Canvas and motion code cannot use var(--token) directly, so they read resolved token values.
// Re-read on mount and on every prototype-param event (harmless while only one scheme exists).
const THEME_EVENT = 'prototype-param';

export type RGB = [number, number, number];

/** Resolved value of a CSS custom property on :root (e.g. '--accent'). */
export function readToken(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

/** Parses #rgb, #rrggbb or rgb(r, g, b) token values into 0-255 channels. */
export function parseRgb(value: string): RGB {
  const v = value.trim();
  if (v.startsWith('#')) {
    const hex = v.length === 4 ? [...v.slice(1)].map((c) => c + c).join('') : v.slice(1);
    return [0, 2, 4].map((i) => parseInt(hex.slice(i, i + 2), 16)) as RGB;
  }
  const m = v.match(/rgba?\(\s*([\d.]+)[\s,]+([\d.]+)[\s,]+([\d.]+)/);
  if (m) return [Number(m[1]), Number(m[2]), Number(m[3])];
  throw new Error(`Unsupported colour token value: ${value}`);
}

/** Resolved token values for the current theme. Pass a stable (module-level) array of names. */
export function useThemeTokens<T extends string>(names: readonly T[]): Record<T, string> {
  const [version, setVersion] = useState(0);

  useEffect(() => {
    // rAF: read after the current frame so styles have settled before we read.
    const bump = () => requestAnimationFrame(() => setVersion((n) => n + 1));
    bump();
    window.addEventListener(THEME_EVENT, bump);
    return () => window.removeEventListener(THEME_EVENT, bump);
  }, []);

  return useMemo(
    () => Object.fromEntries(names.map((n) => [n, readToken(n)])) as Record<T, string>,
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [version, names.join('|')],
  );
}

import { useSyncExternalStore } from 'react';

/** Breakpoint below which the sidebar becomes a drawer. Matches the @media rules in styles.css. */
export const NARROW_QUERY = '(max-width: 899px)';

/** True while the media query matches, re-rendering when it changes. */
export function useMediaQuery(query: string): boolean {
  return useSyncExternalStore(
    (onChange) => {
      const mql = window.matchMedia(query);
      mql.addEventListener('change', onChange);
      return () => mql.removeEventListener('change', onChange);
    },
    () => window.matchMedia(query).matches,
    () => false,
  );
}

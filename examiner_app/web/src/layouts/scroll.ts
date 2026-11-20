import { useEffect, useState } from 'react';

/** Scroll-spy: id of the section currently crossing the reading band, or the first one. */
export function useScrollSpy(ids: string[]): string | undefined {
  const [active, setActive] = useState<string | undefined>(ids[0]);
  const key = ids.join('|');

  useEffect(() => {
    const order = key.split('|');
    const visible = new Set<string>();
    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) visible.add(entry.target.id);
          else visible.delete(entry.target.id);
        }
        const first = order.find((id) => visible.has(`sec-${id}`));
        if (first) setActive(first);
      },
      { rootMargin: '-35% 0px -55% 0px' },
    );
    for (const id of order) {
      const el = document.getElementById(`sec-${id}`);
      if (el) io.observe(el);
    }
    return () => io.disconnect();
  }, [key]);

  return active;
}

export function scrollToSection(id: string, reduced: boolean | null): void {
  document
    .getElementById(`sec-${id}`)
    ?.scrollIntoView({ behavior: reduced ? 'auto' : 'smooth', block: 'start' });
}


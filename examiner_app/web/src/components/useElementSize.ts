import { useEffect, useState } from 'react';

export interface ElementSize {
  width: number;
  height: number;
}

/**
 * Content-box size of an element, kept current with a ResizeObserver. Attach the returned callback ref.
 * Reads 0 x 0 until the first measurement, so callers should wait for a non-zero size before drawing.
 */
export function useElementSize<T extends HTMLElement = HTMLDivElement>(): [(node: T | null) => void, ElementSize] {
  const [node, setNode] = useState<T | null>(null);
  const [size, setSize] = useState<ElementSize>({ width: 0, height: 0 });

  useEffect(() => {
    if (!node) return;
    const apply = (width: number, height: number) => {
      const w = Math.floor(width);
      const h = Math.floor(height);
      setSize((prev) => (prev.width === w && prev.height === h ? prev : { width: w, height: h }));
    };
    const rect = node.getBoundingClientRect();
    apply(rect.width, rect.height);
    const observer = new ResizeObserver((entries) => {
      const box = entries[0]?.contentRect;
      if (box) apply(box.width, box.height);
    });
    observer.observe(node);
    return () => observer.disconnect();
  }, [node]);

  return [setNode, size];
}

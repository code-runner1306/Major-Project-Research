import { useEffect, useId, useRef, useState, type ReactNode } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import './story.css';

/**
 * Collapsed "More detail" button. Opening it slides a drawer over the right side of the current slide,
 * which scrolls inside itself. Nothing below the button moves. Escape or Close returns focus to the button.
 */
export function MoreDetail({ children, label = 'More detail' }: { children: ReactNode; label?: string }) {
  const [open, setOpen] = useState(false);
  const reduce = useReducedMotion() ?? false;
  const drawerId = useId();
  const toggleRef = useRef<HTMLButtonElement>(null);
  const closeRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!open) return;
    closeRef.current?.focus();
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        e.stopPropagation();
        setOpen(false);
        toggleRef.current?.focus();
      }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [open]);

  const close = () => {
    setOpen(false);
    toggleRef.current?.focus();
  };

  return (
    <div className="more-detail">
      <button
        ref={toggleRef}
        type="button"
        className="more-detail__toggle"
        aria-expanded={open}
        aria-controls={drawerId}
        onClick={() => setOpen((o) => !o)}
      >
        <span className="more-detail__mark" aria-hidden="true">
          {open ? '−' : '+'}
        </span>
        {label}
      </button>
      <AnimatePresence initial={false}>
        {open && (
          <motion.aside
            key="drawer"
            id={drawerId}
            className="more-detail__drawer"
            aria-label={label}
            initial={reduce ? false : { x: '100%', opacity: 0 }}
            animate={{ x: 0, opacity: 1 }}
            exit={reduce ? { opacity: 0 } : { x: '100%', opacity: 0 }}
            transition={reduce ? { duration: 0 } : { duration: 0.25, ease: 'easeOut' }}
          >
            <div className="more-detail__head">
              <span className="more-detail__title">{label}</span>
              <button ref={closeRef} type="button" className="more-detail__close" onClick={close}>
                Close
              </button>
            </div>
            <div className="more-detail__inner">{children}</div>
          </motion.aside>
        )}
      </AnimatePresence>
    </div>
  );
}

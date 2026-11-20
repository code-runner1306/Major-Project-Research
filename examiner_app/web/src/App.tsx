import { useEffect, useState } from 'react';
import { AnimatePresence, motion, useReducedMotion } from 'motion/react';
import { findExperiment, EXPERIMENTS } from './experiments';
import { useHashRoute, DEFAULT_ROUTE } from './shell/useHashRoute';
import { Sidebar } from './shell/Sidebar';
import { TopBar } from './shell/TopBar';
import { PlaceholderPage } from './shell/PlaceholderPage';
import { PAGES } from './pages';
import { SectionErrorBoundary } from './components/SectionErrorBoundary';

export default function App() {
  const routeId = useHashRoute().id;
  const def = findExperiment(routeId) ?? findExperiment(DEFAULT_ROUTE) ?? EXPERIMENTS[0];
  const reduced = useReducedMotion();
  const offset = reduced ? 0 : 8;
  // Narrow screens only: the experiment drawer is closed whenever the experiment changes.
  const [navOpen, setNavOpen] = useState(false);

  useEffect(() => {
    setNavOpen(false);
  }, [def.id]);

  // Start each experiment at the top of the page.
  useEffect(() => {
    window.scrollTo({ top: 0, behavior: 'auto' });
  }, [def.id]);

  const Page = PAGES[def.id];

  return (
    <div className="app">
      <TopBar open={navOpen} onToggle={() => setNavOpen((o) => !o)} />
      <Sidebar activeId={def.id} open={navOpen} onClose={() => setNavOpen(false)} />
      <main className="content">
        <AnimatePresence mode="wait" initial={false}>
          <motion.div
            key={def.id}
            className="page"
            initial={{ opacity: 0, y: offset }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -offset }}
            transition={{ duration: reduced ? 0 : 0.2 }}
          >
            <SectionErrorBoundary
              fallback={() => (
                <div className="section-error page-error" role="alert">
                  <p>
                    Something went wrong while showing this page. The rest of the app still works.
                  </p>
                  <p>
                    <a href="#/intro">Go to Introduction</a>
                  </p>
                </div>
              )}
            >
              {Page ? <Page def={def} /> : <PlaceholderPage def={def} />}
            </SectionErrorBoundary>
          </motion.div>
        </AnimatePresence>
      </main>
    </div>
  );
}

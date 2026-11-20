import { motion, useReducedMotion } from 'motion/react';
import { Term } from '../../components/Term';

const COLUMNS = 32;

/**
 * Two memory grids. Each column is one key coordinate of the recurrent state. Scalar decay keeps one shared
 * fraction for every column at every token. Channel-wise decay keeps a different fraction per column. The
 * channel-wise heights are the saved per-channel averages of a trained channel-wise model (no invented values).
 */
export function DecayDiagram({ channelValues }: { channelValues: number[] | null }) {
  const reduce = useReducedMotion() ?? false;
  const values = channelValues ?? Array.from({ length: COLUMNS }, () => 0);
  const scalarLevel = 0.5; // drawn as a plain shared bar; no measured number is shown

  return (
    <div className="anat-decay" role="img" aria-label="Scalar decay shares one kept fraction across all memory columns. Channel-wise decay gives each column its own.">
      <figure className="anat-decay__panel">
        <figcaption>
          <strong>Scalar decay</strong>
          <span>one value per token, shared by every column</span>
        </figcaption>
        <div className="anat-decay__grid">
          {Array.from({ length: COLUMNS }, (_, i) => (
            <div key={i} className="anat-decay__col anat-decay__col--shared">
              <motion.div
                className="anat-decay__bar"
                initial={false}
                animate={{ height: `${scalarLevel * 100}%` }}
                transition={reduce ? { duration: 0 } : { duration: 0.6, delay: 0.01 * i }}
              />
            </div>
          ))}
        </div>
        <p className="anat-decay__axis">memory columns (key coordinates) →</p>
      </figure>

      <figure className="anat-decay__panel">
        <figcaption>
          <strong>
            <Term term="channel-wise-decay">Channel-wise decay</Term>
          </strong>
          <span>one value per column</span>
        </figcaption>
        <div className="anat-decay__grid">
          {values.map((v, i) => (
            <div key={i} className="anat-decay__col">
              <motion.div
                className="anat-decay__bar anat-decay__bar--own"
                initial={reduce ? false : { height: '0%' }}
                animate={{ height: `${Math.round(v * 1000) / 10}%` }}
                transition={reduce ? { duration: 0 } : { duration: 0.7, delay: 0.015 * i, ease: 'easeOut' }}
              />
            </div>
          ))}
        </div>
        <p className="anat-decay__axis">
          {channelValues ? 'saved averages from a trained model' : 'saved averages unavailable'} →
        </p>
      </figure>
    </div>
  );
}

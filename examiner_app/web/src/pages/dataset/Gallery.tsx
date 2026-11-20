import { WhatThisShows } from '../../components/WhatThisShows';
import {
  ExampleError,
  OriginBadge,
  ScoredStrip,
  Skeleton,
  WaitingForServer,
  defaultParams,
  useExample,
  useGenerators,
} from './shared';
import type { GeneratorInfo } from './types';

// One plain sentence per generator, using only the facts in content_briefs/dataset.md.
const BLURB: Record<string, string> = {
  mqar: 'Four key-value pairs first, then filler, then each key asked once. Four scored answers, one per asked key.',
  overwrite: 'One key is written several times. The model must return its latest value and the value of an unrelated key.',
  arm: 'The overwrite example used for the training arms. The arms change the query distance (8 to 16) or the number of rewrites (1 to 3).',
  layout: 'An overwrite example, drawn until its write pattern matches a named layout.',
  length: 'The same Zoology memory at 64, 256 or 1024 tokens. Filler makes it longer; the memory itself does not change.',
  kata: 'Two passes write every key, then each key is asked once. The scored answers are the latest values. No filler.',
  anatomy_e1: 'Keys sit at fixed even positions. One query key is followed by a masked slot, the answer to fill in. The rest is random noise.',
  anatomy_e4: 'A four-pair table, a 48-token random haystack, then four queries of the form key then mask.',
  anatomy_e5: 'Like E4, but the haystack is full of decoy pairs that reuse the table keys with wrong values. The true answer is the first binding.',
  zoology_cell: 'Zoology MQAR with the memory shifted by 1 to 16 tokens, or with the queries near, far, in memory order or reversed.',
  tinystories:
    'A short children’s story from the public TinyStories collection. Each word piece is shown as the GPT-2 tokenizer split it, cut to a window of 64 tokens. The end-of-text marker appears only if the story fits inside the window.',
};

/** TinyStories is always requested live, with the 64-token window; a server-side failure shows as an error. */
const STORY_WINDOW = 64;

function GalleryCard({ info }: { info: GeneratorInfo }) {
  const isStory = info.id === 'tinystories';
  const wantsLive = info.live || isStory;
  const params = isStory ? { ...defaultParams(info), window: STORY_WINDOW } : defaultParams(info);
  const ex = useExample(wantsLive ? { generator: info.id, seed: 0, params } : null, 0);

  return (
    <article className="ds-card">
      <header className="ds-card__head">
        <h3 className="ds-card__name">{info.name}</h3>
        <OriginBadge origin={info.origin} />
      </header>
      <p className="ds-card__what">{BLURB[info.id] ?? info.source}</p>

      {!wantsLive && (
        <p className="ds-card__static">
          This one is shown as a description only: it needs data that is not on this machine.
        </p>
      )}
      {wantsLive && ex.error && <ExampleError error={ex.error} />}
      {wantsLive && ex.waiting && <WaitingForServer />}
      {wantsLive && !ex.data && !ex.error && !ex.waiting && <Skeleton label={`loading ${info.name} example`} />}
      {ex.data && (
        <>
          <ScoredStrip ex={ex.data} pending={ex.pending} />
          <p className="ds-card__scored">
            Scored positions:{' '}
            {ex.data.scored.length === 0
              ? 'none'
              : ex.data.scored
                  .slice(0, 6)
                  .map((s) => s.position)
                  .join(', ') + (ex.data.scored.length > 6 ? `, and ${ex.data.scored.length - 6} more` : '')}
          </p>
          <WhatThisShows>
            each square is one token, coloured by its role. Outlined squares are the scored answers: the only positions the model is graded on.
          </WhatThisShows>
          {ex.data.notes.length > 0 && (
            <details className="ds-card__notes">
              <summary>Read what this example says</summary>
              <ul className="ds-list">
                {ex.data.notes.map((note, i) => (
                  <li key={i}>{note}</li>
                ))}
              </ul>
            </details>
          )}
        </>
      )}
    </article>
  );
}

/** One live example per generator, side by side. Each card loads on its own. */
export function Gallery() {
  const gens = useGenerators();
  if (gens.pending) {
    return (
      <div className="ds-cards">
        <Skeleton label="loading the generator list" tall />
        <Skeleton label="loading the generator list" tall />
      </div>
    );
  }
  if (gens.waiting) return <WaitingForServer />;
  if (gens.error) return <ExampleError error={gens.error} what="The dataset list could not be loaded." />;
  return (
    <div className="ds-cards">
      {(gens.data ?? []).map((info) => (
        <GalleryCard key={info.id} info={info} />
      ))}
    </div>
  );
}

import { WhatThisShows } from '../../components/WhatThisShows';
import { OriginBadge } from './shared';
import type { Origin } from './types';

interface Provenance {
  id: string;
  name: string;
  origin: Origin;
  source: string;
  where: string;
}

// Facts from content_briefs/dataset.md (table "Generators"). Nothing here is invented: where a
// source is a paper or repository, its identifier is given as the brief gives it.
const CARDS: Provenance[] = [
  {
    id: 'mqar',
    name: 'Multi-query associative recall (Zoology)',
    origin: 'port',
    source: 'recipe_data.mqar',
    where:
      'Byte-exact port of HazyResearch Zoology, commit de4e258. Checked equal to the original on 1,000 rows.',
  },
  {
    id: 'overwrite',
    name: 'Keyed overwrite (our task)',
    origin: 'ours',
    source: "retrieval.RetrievalDataset, generator v4, via recipe_data.build_data('overwrite')",
    where: 'Written here. Motivated by the repeated-key test in KATA.',
  },
  {
    id: 'arm',
    name: 'Layout-fix training arms A0 to A4',
    origin: 'ours',
    source: 'layout_fix_data.build_arm_data',
    where: 'Written here, as an adaptation of overwrite v4.',
  },
  {
    id: 'layout',
    name: 'Named test layouts',
    origin: 'ours',
    source: 'confirm_block_layout.sample_layouts and special_patterns',
    where:
      'Written here. Overwrite examples are drawn until their write pattern matches a named layout.',
  },
  {
    id: 'length',
    name: 'Longer sequences (filler inserted)',
    origin: 'ours',
    source: 'run_paper_mqar_length_extension.py logic',
    where:
      'Written here. Zoology MQAR with filler inserted after the 8 memory tokens. The memory stays the same; the queries move later.',
  },
  {
    id: 'kata',
    name: 'KATA two-pass overwrite',
    origin: 'ours',
    source: 'kata_overwrite_data.generate',
    where:
      'Our reconstruction of Ghriss and Chakraborty (arXiv 2607.17419), Appendix F. Their data code is not published.',
  },
  {
    id: 'anatomy_e1',
    name: 'Anatomy E1: masked recall',
    origin: 'upstream',
    source: 'third_party/dualgoose/dualgoose/toy_data.make_mqar_batch',
    where:
      'Upstream code, JIBSIL/dualgoose commit 8ca624a. From Boesch and Wee (arXiv 2609.16183).',
  },
  {
    id: 'anatomy_e4',
    name: 'Anatomy E4: distractor haystack',
    origin: 'upstream',
    source: 'third_party/dualgoose/dualgoose/toy_data.make_long_retrieval_batch (collision off)',
    where:
      'Upstream code, JIBSIL/dualgoose commit 8ca624a. From Boesch and Wee (arXiv 2609.16183).',
  },
  {
    id: 'anatomy_e5',
    name: 'Anatomy E5: colliding keys',
    origin: 'upstream',
    source: 'third_party/dualgoose/dualgoose/toy_data.make_long_retrieval_batch (collision on)',
    where:
      'Upstream code, JIBSIL/dualgoose commit 8ca624a. From Boesch and Wee (arXiv 2609.16183).',
  },
  {
    id: 'zoology_cell',
    name: 'Zoology layout test cells',
    origin: 'ours',
    source: 'zoology_layout_test.make_cell',
    where: 'Written here, built on the Zoology port.',
  },
  {
    id: 'tinystories',
    name: 'TinyStories (language)',
    origin: 'external',
    source: 'Hugging Face roneneldan/TinyStories (validation split), GPT-2 tokenizer, cut to a window of 64 to 512 tokens',
    where:
      'External data, read from a local copy of the validation split and the GPT-2 tokenizer. Stories are generated live here, one per seed.',
  },
];

/** One card per generator: where its code came from, with the badge the brief assigns. */
export function ProvenanceCards() {
  return (
    <figure className="ds-cards-wrap">
      <div className="ds-cards">
        {CARDS.map((card) => (
          <article className="ds-card" key={card.id}>
            <header className="ds-card__head">
              <h3 className="ds-card__name">{card.name}</h3>
              <OriginBadge origin={card.origin} />
            </header>
            <p className="ds-card__source">
              <code>{card.source}</code>
            </p>
            <p className="ds-card__what">{card.where}</p>
          </article>
        ))}
      </div>
      <WhatThisShows>
        each card is one generator. Ported means copied from a published repository and checked to match it. Ours means written in this project.
        Upstream means the authors&apos; code, run as published. External means outside data.
      </WhatThisShows>
    </figure>
  );
}

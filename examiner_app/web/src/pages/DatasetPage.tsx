import type { ReactNode } from 'react';
import type { CodeStep, PageProps, Section } from '../types';
import { ExperimentPage } from '../layouts/ExperimentPage';
import { CodeWalkthrough } from '../components/CodeWalkthrough';
import { Term } from '../components/Term';
import { WhatThisShows } from '../components/WhatThisShows';
import { ExampleError, useGenerators } from './dataset/shared';
import { FamilyTree } from './dataset/FamilyTree';
import { ProvenanceCards } from './dataset/ProvenanceCards';
import { Gallery } from './dataset/Gallery';
import { BuildAssembly } from './dataset/BuildAssembly';
import { GenerateYourOwn } from './dataset/GenerateYourOwn';
import './dataset/dataset.css';

// Code steps: each step's notes match a line in the real source (checked against the files named here).
const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Zoology port',
    plain:
      'The Zoology MQAR generator, ported byte for byte. It draws the keys and values, places the memory at the start, and spaces the queries with a power-law rule.',
    path: 'recipe_data.py',
    symbol: 'mqar',
    notes: [
      { match: 'rng = np.random.RandomState(seed)', note: 'A private random generator, seeded the same way as the original, so the same seed gives the same rows.' },
      { match: 'key_choices = np.arange(1,vocabulary//2)', note: 'Keys come from the lower half of the vocabulary and values from the upper half, so a key is never confused with a value.' },
      { match: 'power=.01*np.arange', note: 'The gaps before each query follow a power-law distribution: short gaps are more likely, but long ones still happen.' },
      { match: 'np.put_along_axis(queries,gaps*2,keys', note: 'Each key is asked once, at its chosen position in the query part.' },
      { match: 'np.put_along_axis(labels,gaps*2+context+1,values', note: 'The scored answer for each query is the value stored with that key.' },
    ],
  },
  {
    title: '2 Overwrite (ours)',
    plain:
      'Our overwrite generator. One key is written several times, and the model must return its latest value. This is the task behind the layout-fix study.',
    path: 'retrieval.py',
    symbol: 'RetrievalDataset._overwrite_v4',
    notes: [
      { match: 'target_index = rng.randrange(key_count)', note: 'Picks the key that will be written again.' },
      { match: 'rewrites = rng.sample(value_pool, overwrite_count)', note: 'New values for the rewrites. Each rewrite gets a value that no other write uses.' },
      { match: 'unrelated_writes = [', note: 'The other keys are written too, so the latest write has to be found among them.' },
      { match: 'minimum_index = max(', note: 'The latest write must come after every earlier write of the key, and far enough back that the query distance fits.' },
      { match: 'maximum_index = min(', note: 'At least one unrelated write must come after the latest write.' },
    ],
  },
  {
    title: '3 The one line that differs',
    plain:
      'The five training-data arms. Each arm changes one or two settings in this table. The model, the seeds and the starting weights are shared by all arms.',
    path: 'layout_fix_data.py',
    lines: { start: 20, end: 27 },
    notes: [
      { match: "'A0_control'", note: 'Control: the query always comes 8 tokens after the memory, and there is always exactly one rewrite.' },
      { match: "'A1_varied_distance'", note: 'The query distance varies from 8 to 16 tokens, so the answer is not always in the same place.' },
      { match: "'A2_no_block_first'", note: 'reject_block_first=True throws away block-first layouts, so no training example has every write of the queried key first.' },
      { match: "'A3_mixed_rewrites'", note: 'The number of rewrites varies from 1 to 3, so the latest value is not always the second write.' },
      { match: "'A4_distance_and_rewrites'", note: 'Both changes at once.' },
    ],
  },
  {
    title: '4 KATA (reconstructed)',
    plain:
      'Our reconstruction of the KATA overwrite test. Two full passes write every key, then each key is asked once. There is no filler.',
    path: 'kata_overwrite_data.py',
    symbol: 'generate',
    notes: [
      { match: 'order = rng.permutation(keys)', note: 'Each pass writes the keys in a new random order.' },
      { match: 'tokens[row, cursor], tokens[row, cursor + 1] = key_tokens[k], values[p, k]', note: 'Each write is a key token followed by its value token.' },
      { match: 'tokens[row, cursor] = key_tokens[k]', note: 'Each key is then asked once: the key goes in, and the slot after it is left for the answer.' },
      { match: 'labels[row, cursor] = latest', note: 'The scored answer is the latest value written for that key.' },
    ],
  },
  {
    title: '5 Anatomy (upstream)',
    plain:
      'Upstream code from the Anatomy harness, run as published. It builds a table of pairs, then a haystack, then queries. The collision switch is what turns E4 into E5.',
    path: 'third_party/dualgoose/dualgoose/toy_data.py',
    symbol: 'make_long_retrieval_batch',
    notes: [
      { match: 'clean = _randint(val_lo, vocab_size - 1', note: 'The sequence starts as random distractor tokens; the table and the queries are written over it.' },
      { match: 'table_start = seq_len - tail - gap - 2 * pairs', note: 'Where the table starts. The gap sets the distance between the table and the queries.' },
      { match: 'if collision:', note: 'The collision switch (E5): the haystack is filled with decoy pairs that reuse the table keys.' },
      { match: 'clean[:, gap_start + 2 * d] = key_lo + dk', note: 'Each decoy reuses one of the table keys with a wrong value.' },
    ],
  },
];

/** Dataset page: families, provenance, code, a gallery, one example built three ways, and a generator to try. */
export function DatasetPage({ def }: PageProps) {
  const gens = useGenerators();
  const notice: ReactNode = gens.error ? (
    <ExampleError error={gens.error} what="The dataset list could not be loaded." />
  ) : undefined;

  const sections: Section[] = [
    {
      id: 'question',
      title: 'Question',
      explain: (
        <>
          <p className="ds-lead">What data did the models learn from and get tested on?</p>
          <p>
            The tests in this project are built by small programs that each write a sequence of symbols. Each symbol is a{' '}
            <Term term="token" />: a single unit the model reads one at a time. A <Term term="mqar">MQAR</Term> example, for instance, is a list
            of <Term term="key" />-and-<Term term="value" /> pairs, followed by questions asking for the value of a key.
          </p>
          <p>
            Eleven generators come from four families: the Zoology benchmark, a paper called KATA, the Anatomy harness, and the public TinyStories
            text. Some are our code, some we ported, some are the authors&apos; code, and TinyStories is outside public data, read from a local copy.
          </p>
          <p className="ds-muted">
            Different generators use different reserved tokens. A 0 can be a query marker in one and filler in another.
          </p>
        </>
      ),
      visual: <FamilyTree />,
    },
    {
      id: 'model',
      title: 'Where each dataset comes from',
      explain: (
        <>
          <p>
            Each card names the code behind one generator and where it came from. A ported generator was copied from a published repository and
            checked to match it. Our generators are written in this project. The Anatomy generators are the authors&apos; own code.
          </p>
          <p>
            The point of this section is trust: when a result depends on a dataset, you can see whether that dataset is the published one or our own
            variant of it.
          </p>
        </>
      ),
      visual: <ProvenanceCards />,
    },
    {
      id: 'code',
      title: 'Code',
      explain: (
        <>
          <p>
            Follow five generators from the code that writes them. Each step shows real lines from the repository. Highlighted lines have a note:
            hover or click one to read it.
          </p>
          <p>
            The Zoology port is the reference. Our overwrite task is the one the layout-fix study trains on. The Anatomy code is kept exactly as its
            authors wrote it.
          </p>
        </>
      ),
      visual: (
        <>
          <CodeWalkthrough steps={CODE_STEPS} />
          <WhatThisShows>the real source, read from the repository when you open this page, with notes on the lines that matter.</WhatThisShows>
        </>
      ),
    },
    {
      id: 'results',
      title: 'Example gallery',
      explain: (
        <>
          <p><strong>Main takeaway:</strong> These examples show what each task asks a model to store, update, and retrieve; performance comparisons appear in the experiment results.</p>
          <p>
            One live example per generator, produced by the same code the experiments use. Each strip uses one colour per role, and the legend under
            it names them.
          </p>
          <p>
            Outlined tokens are the <strong>scored answers</strong>: the only positions where the model is graded. The count differs by test: MQAR
            scores four answers, the overwrite task two, and KATA one per key.
          </p>
        </>
      ),
      visual: <Gallery />,
    },
    {
      id: 'inside',
      title: 'How one example is built',
      explain: (
        <>
          <p>
            Here one <Term term="overwrite-task" /> example is built step by step, the way the generator builds it. Each step draws a piece and
            places it in the example: first the keys, then their values, then a rewrite of one key, then the filler and the queries. Every piece
            is a real token from the server, placed where the generator put it.
          </p>
        </>
      ),
      visual: <BuildAssembly />,
    },
    {
      id: 'try',
      title: 'Generate your own',
      explain: (
        <>
          <p>
            Pick a generator, set its parameters, and roll the seed for a new example. The reading under the strip is written from the example
            itself. The raw JSON is there for the exact numbers.
          </p>
          <p className="ds-muted">
            TinyStories is generated live here too: pick a window of 64, 128, 256 or 512 tokens. Its story text appears above the strip, and each
            chip shows the word piece the tokenizer produced (hover for the token id).
          </p>
        </>
      ),
      visual: <GenerateYourOwn />,
    },
  ];

  return <ExperimentPage def={def} sections={sections} notice={notice} />;
}

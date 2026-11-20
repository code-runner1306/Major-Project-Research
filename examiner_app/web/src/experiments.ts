import type { ExperimentDef } from './types';

// Registry of every experiment page, in the order the sidebar shows them. See ../../SPEC.md "Routing and shell".
export const EXPERIMENTS: ExperimentDef[] = [
  // Start
  {
    id: 'intro',
    title: 'Introduction',
    group: 'start',
    status: 'live',
    oneLiner:
      'Does giving each memory column its own fading speed help a small language model remember?',
    analogy:
      'A notebook too small for every page: old notes fade at one speed, or at a speed chosen per column.',
  },
  {
    id: 'dataset',
    title: 'The datasets',
    group: 'start',
    status: 'live',
    oneLiner:
      'The synthetic and public text the models are trained and tested on, and how each test is built.',
    analogy:
      'Like the question papers and practice sets a teacher writes: the same kind of test has to be built carefully so that the results mean what they seem to mean.',
  },

  // Experiments
  {
    id: 'layout-fix',
    title: 'Layout fix A0–A4',
    group: 'experiments',
    status: 'live',
    oneLiner:
      'Models trained the usual way scored 97–100% on ordinary validation tests, yet failed a special "block-first" layout by returning the first rewrite of a value instead of the latest one.',
    analogy:
      'Like a student who gets full marks on the practice paper but, when an answer is crossed out and rewritten mid-exam, copies down the first attempt. Changing the kind of practice they get fixes the habit.',
    headline: {
      value: '6 / 6',
      label: 'models trained on the A4 data that no longer fail the block-first layout',
    },
  },
  {
    id: 'shortcut',
    title: 'The hidden shortcut',
    group: 'experiments',
    status: 'live',
    oneLiner: 'Some models pass ordinary tests by copying a position in memory, not the right value.',
    analogy: 'Like a student who always copies the second answer on the sheet and passes every practice quiz.',
  },
  {
    id: 'decay-init',
    title: 'Decay initialization',
    group: 'experiments',
    status: 'live',
    oneLiner:
      'Asks whether the way the forgetting rates are set before training changes how the two decay designs compare.',
    analogy:
      'Like choosing where to start a race: the starting position might change who looks fastest at the finish, or it might not.',
  },
  {
    id: 'stress',
    title: 'Stress tests and gate interventions',
    group: 'experiments',
    status: 'live',
    oneLiner:
      'Asks how the models hold up when the tests get harder, and what changes when the forgetting gates are overridden by hand.',
    analogy:
      'Like taking a car to a steep hill or a rough road and then disconnecting one part to see what it was doing.',
  },
  {
    id: 'length',
    title: 'Longer sequences and filler decay',
    group: 'experiments',
    status: 'live',
    oneLiner:
      'Asks how the models behave on longer inputs, and what happens to stored memory across long stretches of filler text.',
    analogy:
      'Like reading a long report with many irrelevant paragraphs in between: the question is how much of the early part is still remembered by the end.',
  },
  {
    id: 'anatomy',
    title: 'Anatomy harness',
    group: 'experiments',
    status: 'results',
    oneLiner:
      "Asks whether the scalar vs channel-wise result holds in someone else's published experiments: we re-ran the Anatomy of Associative Recall code exactly, then added a scalar-decay model to it.",
    analogy:
      "Like re-running another lab's published recipe with their own ingredients, then changing just one ingredient to see if the dish changes.",
  },
  {
    id: 'controls',
    title: 'Negative controls (Zoology, KATA)',
    group: 'experiments',
    status: 'live',
    oneLiner:
      'Asks whether two external benchmarks, Zoology and KATA, can tell the decay designs apart, as a check on the method.',
    analogy:
      'Like testing a thermometer in a bath of known temperature before trusting it with a new reading.',
  },
  {
    id: 'tinystories',
    title: 'TinyStories language models',
    group: 'experiments',
    status: 'live',
    oneLiner:
      'Asks whether the decay designs make a difference when the models learn from short, simple stories instead of synthetic tests.',
    analogy:
      'Like moving from practice drills to reading ordinary books: the question is whether what was learned in drills carries over.',
  },
  {
    id: 'cost',
    title: 'Cost and runtime',
    group: 'experiments',
    status: 'results',
    oneLiner:
      'Asks how much time and computing the two decay designs need to train and to run.',
    analogy:
      'Like comparing two engines on fuel use and top speed, not only on how far they get.',
  },

  // Appendix
  {
    id: 'selective-reset',
    title: 'Selective reset pilot',
    group: 'appendix',
    status: 'live',
    oneLiner:
      'A small pilot asking whether a model can learn to clear only part of its memory when it needs to.',
    analogy:
      'Like a whiteboard where you can erase one line without wiping the rest.',
  },
  {
    id: 'attention-control',
    title: 'Attention control',
    group: 'appendix',
    status: 'results',
    oneLiner:
      'Asks how a standard attention-based model performs on the same tests, as a point of comparison.',
    analogy:
      'Like a second, different car run on the same course, so the first car\'s times have something to be measured against.',
  },
  {
    id: 'decision',
    title: 'Decision fine-tuning',
    group: 'appendix',
    status: 'results',
    oneLiner:
      'Asks whether a short extra round of training on a decision task changes which answer the models pick.',
    analogy:
      'Like a short refresher course after someone has already finished their training, to see whether their choices shift.',
  },
];

export function findExperiment(id: string): ExperimentDef | undefined {
  return EXPERIMENTS.find((e) => e.id === id);
}

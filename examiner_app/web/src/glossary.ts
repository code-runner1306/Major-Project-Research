// Plain-English definitions used by the <Term> component. Keys are lowercase slugs.
export const GLOSSARY: Record<string, { term: string; plain: string }> = {
  // Research organisation (from GLOSSARY.md)
  experiment: {
    term: 'Experiment',
    plain:
      'One research question with its own result. An experiment can use several scripts and several trained models.',
  },
  run: {
    term: 'Run',
    plain:
      'One trained model, defined by its training condition, its decay design and its random seed.',
  },
  arm: {
    term: 'Arm',
    plain:
      'One training-data condition inside the layout-fix experiment, labelled A0 (the control) to A4.',
  },

  // Model (from GLOSSARY.md)
  'decay-granularity': {
    term: 'Decay granularity',
    plain:
      'How many forgetting rates the model uses: one shared rate for each step, or a separate rate for each column of its memory. This is the only design difference being compared.',
  },
  'scalar-decay': {
    term: 'Scalar decay',
    plain:
      'The design where each step gets one forgetting rate, and that single rate fades the whole memory at once.',
  },
  'channel-wise-decay': {
    term: 'Channel-wise decay',
    plain:
      'The design where each step gets a separate forgetting rate for every column of the memory, so different columns can fade at different speeds.',
  },
  'decay-gate': {
    term: 'Decay gate (α)',
    plain:
      "The model\'s learned forgetting rate, a number between 0 and 1, that shrinks the old memory before each new write. Closer to 1 means the memory is kept for longer.",
  },
  'write-gate': {
    term: 'Write gate (β)',
    plain:
      'A learned number that sets how strongly each new value overwrites what the memory already holds for its key. It works the same way in both designs.',
  },
  'recurrent-state': {
    term: 'Recurrent state',
    plain:
      "The fixed-size table the model keeps and updates as it reads each token. It is the model\'s whole memory of what it has read so far.",
  },
  'decay-initialization': {
    term: 'Decay initialization',
    plain:
      'How the forgetting rates are set before training starts. The study compares a default setting, a mean-decay setting and a joint setting.',
  },

  // Extra terms requested for the app
  token: {
    term: 'Token',
    plain:
      'A small unit of input that the model reads one at a time. Here each token is a single symbol from a fixed list.',
  },
  key: {
    term: 'Key',
    plain:
      'A label the model uses to look something up, like the tab on a filing-cabinet folder.',
  },
  value: {
    term: 'Value',
    plain:
      'The content stored under a key, like the paper kept inside that folder.',
  },
  query: {
    term: 'Query',
    plain:
      'A question near the end of the input that asks the model to return the value stored under a given key.',
  },
  'filler-token': {
    term: 'Filler token',
    plain:
      'A neutral placeholder token that carries no information. Filler is used to pad the input so that it gets longer.',
  },
  mqar: {
    term: 'MQAR (multi-query associative recall)',
    plain:
      'A test where several key-and-value pairs are shown first, then the model is asked for the value of several keys in turn and must get each one right.',
  },
  'overwrite-task': {
    term: 'Overwrite task',
    plain:
      'A test where the same key is written several times with different values, and the model must return the most recent one.',
  },
  'delta-rule': {
    term: 'Delta rule',
    plain:
      'A way of updating memory. The model first checks what its memory currently holds for a key, then moves that entry toward the new value by only the difference.',
  },
  'gated-deltanet': {
    term: 'Gated DeltaNet',
    plain:
      'A small sequence model whose memory is a fixed-size table. Each step it updates that table with the delta rule and with learned forgetting and write rates.',
  },
  parameter: {
    term: 'Parameter',
    plain:
      'A learned number inside the model. Training adjusts these numbers; a model with about 1.6 million parameters has about 1.6 million of them.',
  },
  checkpoint: {
    term: 'Checkpoint',
    plain:
      "A saved snapshot of a model\'s parameters at one point in training. It can be reloaded to run the model exactly as it was at that point.",
  },
  seed: {
    term: 'Seed',
    plain:
      'The starting number for the random choices made during training and data generation. The same seed repeats a run exactly, and different seeds give independent repeats.',
  },
  epoch: {
    term: 'Epoch',
    plain:
      'One complete pass through the training data.',
  },
  'validation-set': {
    term: 'Validation set',
    plain:
      'A set of examples held back from training and used to check how well the model is doing. The model never trains on these examples.',
  },
  'fresh-data': {
    term: 'Fresh data',
    plain:
      'Newly generated examples that were not used for training or for the saved checks. They test whether a result holds on cases the model has never seen.',
  },
  'block-first-layout': {
    term: 'Block-first layout',
    plain:
      "An arrangement where every write of the asked-about key comes before any other key's write, like writing all of Alice's phone numbers first and everyone else's afterwards. About one in three training examples happen to look like this.",
  },
  shortcut: {
    term: 'Shortcut',
    plain:
      'A rule that gives the right answer on the training data for the wrong reason. Here, models learned "in a block-first layout, answer the second write" instead of "answer the newest value for this key".',
  },
  'half-life': {
    term: 'Half-life',
    plain:
      'How many steps it takes for a memory to lose half of its strength under a fixed forgetting rate. A faster forgetting rate gives a shorter half-life.',
  },
  'lock-in': {
    term: 'Lock-in',
    plain:
      'A training run counts as locked in when it finally solves the task, reaching the accuracy threshold set by the Anatomy paper (90% or 80%, depending on the test).',
  },
  zoology: {
    term: 'Zoology',
    plain:
      'A benchmark suite of small, made-up memory tests, used to compare sequence models on skills such as recalling values by key.',
  },
  kata: {
    term: 'KATA',
    plain:
      'Kernelized Linear Attention, a 2026 paper whose repeated-key overwrite test we re-created. We used it as a negative control: a well-designed dataset where the shortcut should not appear.',
  },
  'anatomy-harness': {
    term: 'Anatomy harness',
    plain:
      'The published code from the paper "Anatomy of Associative Recall in Fixed-State Recurrences". We re-ran its experiments exactly and added a scalar-decay version of its model to compare against.',
  },
  tinystories: {
    term: 'TinyStories',
    plain:
      "A public collection of short, simple children\'s stories, used to train and test small language models on ordinary English text.",
  },

  // Shortcut experiment
  'opening-run-second': {
    term: 'Opening-run second',
    plain:
      'The shortcut rule: if every write of the key comes first in one unbroken run, answer the second write. In training there is only one rewrite, so this always gives the newest value, which is why the models could learn it.',
  },
  'first-replacement': {
    term: 'First replacement',
    plain:
      'The first rewrite of a key, the second value written for it. Most wrong answers in the block-first layout returned this value instead of the newest one.',
  },
  patch: {
    term: 'Patch',
    plain:
      'A direct change to one part of a trained model, such as one gate or one key, made for a test and then undone. It shows which part the answer depends on.',
  },
};

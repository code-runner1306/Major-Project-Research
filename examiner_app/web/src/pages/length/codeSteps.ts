import type { CodeStep } from '../../types';

// Every note's match string is a line (or part of one) in the real source; checked by grep when this file was written.
export const CODE_STEPS: CodeStep[] = [
  {
    title: '1 Lengthen the input',
    plain: 'Extra filler is inserted after the first eight tokens. The labels get nothing at those positions.',
    path: 'run_paper_mqar_length_extension.py',
    symbol: 'run',
    notes: [
      {
        match: 'tokens.new_zeros((EXAMPLES,extra))',
        note: 'The extra positions are token 0, the filler ID the training data already uses. The eight write tokens are kept as they are.',
      },
      {
        match: 'targets.new_full((EXAMPLES,extra),-100)',
        note: 'Filler positions get the ignore label, so only the real query positions are scored.',
      },
    ],
  },
  {
    title: '2 Place the queries',
    plain: 'The four query keys sit at random gaps after the writes. Every other slot stays filler.',
    path: 'recipe_data.py',
    symbol: 'mqar',
    notes: [
      {
        match: 'space=(length-context)//2',
        note: 'The query region is the input after the four writes. Its size sets how far apart the keys can be.',
      },
      {
        match: 'np.put_along_axis(queries,gaps*2,keys,axis=1)',
        note: 'Each query key goes at its own random gap. Slots that are not keys stay zero, which is filler.',
      },
      {
        match: 'np.put_along_axis(labels,gaps*2+context+1,values,axis=1)',
        note: 'The answer is the value written for that key, scored one position after the query key.',
      },
    ],
  },
  {
    title: '3 The decay gate',
    plain: 'Each token gets a forgetting rate between 0 and 1. The stored memory is multiplied by it at every step.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.decay',
    notes: [
      {
        match: 'raw=self.a_proj(x)',
        note: 'The rate depends on the token itself, so filler and content tokens can have different rates.',
      },
      {
        match: 'torch.exp(-self.A_log.float().exp()*F.softplus(raw+self.dt_bias.float()))',
        note: 'The result is the decay gate (α). A value near 1 keeps memory; a value near 0 erases it.',
      },
    ],
  },
];

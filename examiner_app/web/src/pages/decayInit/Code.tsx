import type { CodeStep } from '../../types';

// Every match string is a line (or part of one) in the real source; the server reads the file at request time.
export const CODE_STEPS: CodeStep[] = [
  {
    title: '1 The one line that differs',
    plain: 'The number of decay values per token: one for scalar decay, one per key coordinate for channel-wise decay.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.__init__',
    notes: [
      {
        match: "self.gate_width=1 if variant=='scalar' else self.head_k",
        note: 'Scalar decay: one value per head and token. Channel-wise decay: one value per key coordinate. Nothing else in the layer changes.',
      },
    ],
  },
  {
    title: '2 Default start',
    plain: 'The starting rates and offsets are drawn at random, so each model starts in a different place.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.__init__',
    notes: [
      {
        match: 'rate=torch.empty(heads,1).uniform_(0,16).clamp_min(1e-8)',
        note: 'The rate of forgetting is drawn at random, uniformly between 0 and 16.',
      },
      {
        match: 'dt=torch.exp(torch.rand(heads,1)*(math.log(.1)-math.log(.001))+math.log(.001))',
        note: 'A starting time step is drawn at random, log-uniformly between 0.001 and 0.1.',
      },
      {
        match: 'self.dt_bias=nn.Parameter(inverse.expand(heads,self.gate_width).clone())',
        note: 'The offset that the decay formula starts from. Training adjusts it.',
      },
    ],
  },
  {
    title: '3 Mean-decay start',
    plain: 'Before training, each layer’s offset is fitted so the average decay hits a target half-life: 16, 32, 64 or 128 tokens.',
    path: 'recipe_calibration.py',
    symbol: 'calibrate',
    notes: [
      {
        match: 'target=2**(-1/half_life)',
        note: 'The target average decay for one layer: after its half-life in tokens, half of a value’s weight is left.',
      },
      {
        match: 'fitted=fit_offset(raw,rate,target)',
        note: 'The offset is fitted on the first 256 training examples, so the average decay matches the target.',
      },
      {
        match: 'layer.dt_bias.copy_(fitted.to(device))',
        note: 'Only the offset is replaced. The rates and the projections keep their default values.',
      },
      {
        match: 'if not torch.isfinite(alpha).all() or error>1e-6:',
        note: 'The start is accepted only if every layer’s average is within 0.000001 of its target.',
      },
    ],
  },
  {
    title: '4 The decay formula',
    plain: 'How every token’s decay is computed, the same for both starts. Only the starting values differ.',
    path: 'recipe_model.py',
    symbol: 'RecipeGDN.decay',
    notes: [
      {
        match: 'return torch.exp(-self.A_log.float().exp()*F.softplus(raw+self.dt_bias.float()))',
        note: 'Decay = exp(−rate × softplus(projection + offset)). A larger rate or offset means faster forgetting.',
      },
    ],
  },
];

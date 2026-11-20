import { useState } from 'react';
import { ModelSelect } from './common';
import { SurvivalPanel } from './Survival';
import { MAX_FILLER, sameModel, type LocalModel, type ModelRef } from './types';

/** Inside visual: how much of a memory survives the whole 960-token filler run, layer by layer, on the chosen checkpoint. */
export function LengthInsideVisual({ models }: { models: LocalModel[] }) {
  const preferred: ModelRef = { init: 'mean-decay', variant: 'scalar', seed: 44 };
  const [model, setModel] = useState<ModelRef>(() =>
    models.find((m) => sameModel(m, preferred)) ?? models[0] ?? preferred,
  );
  return (
    <SurvivalPanel
      model={model}
      filler={MAX_FILLER}
      examples={16}
      controls={<ModelSelect models={models} value={model} onChange={setModel} label="Checkpoint" />}
    />
  );
}

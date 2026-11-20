import type { FC } from 'react';
import type { PageProps } from '../types';
import { DatasetPage } from './DatasetPage';
import { IntroPage } from './IntroPage';
import { LayoutFixPage } from './LayoutFixPage';
import { ShortcutPage } from './ShortcutPage';
import { DecayInitPage } from './DecayInitPage';
import { StressPage } from './StressPage';
import { LengthPage } from './LengthPage';
import { AnatomyPage } from './AnatomyPage';
import { ControlsPage } from './ControlsPage';
import { TinyStoriesPage } from './TinyStoriesPage';
import { CostPage } from './CostPage';
import { SelectiveResetPage } from './SelectiveResetPage';
import { AttentionControlPage } from './AttentionControlPage';
import { DecisionPage } from './DecisionPage';

// Written experiment pages, keyed by experiment id. Ids not listed here render PlaceholderPage.
export const PAGES: Record<string, FC<PageProps>> = {
  intro: IntroPage,
  dataset: DatasetPage,
  'layout-fix': LayoutFixPage,
  shortcut: ShortcutPage,
  'decay-init': DecayInitPage,
  stress: StressPage,
  length: LengthPage,
  anatomy: AnatomyPage,
  controls: ControlsPage,
  tinystories: TinyStoriesPage,
  cost: CostPage,
  'selective-reset': SelectiveResetPage,
  'attention-control': AttentionControlPage,
  decision: DecisionPage,
};

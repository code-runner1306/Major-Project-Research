import { WhatThisShows } from '../../components/WhatThisShows';
import { OriginBadge } from './shared';
import type { Origin } from './types';

interface TreeNode {
  id: string;
  name: string;
  origin: Origin;
  note?: string;
  children?: TreeNode[];
}

// Structure from content_briefs/dataset.md: source projects at the top, generators beneath them.
const FAMILIES: TreeNode[] = [
  {
    id: 'zoology',
    name: 'Zoology benchmark',
    origin: 'external',
    note: 'a public suite of memory tests',
    children: [
      {
        id: 'mqar',
        name: 'MQAR, our byte-exact port',
        origin: 'port',
        children: [
          { id: 'length', name: 'Longer sequences (filler added)', origin: 'ours' },
          { id: 'zoology_cell', name: 'Zoology layout test cells', origin: 'ours' },
        ],
      },
    ],
  },
  {
    id: 'kata',
    name: 'KATA paper',
    origin: 'external',
    note: 'a repeated-key overwrite test',
    children: [
      { id: 'kata', name: 'KATA two-pass overwrite (our reconstruction)', origin: 'ours' },
      {
        id: 'overwrite',
        name: 'Keyed overwrite (our v4 task)',
        origin: 'ours',
        children: [
          { id: 'arm', name: 'Training arms A0 to A4', origin: 'ours' },
          { id: 'layout', name: 'Named test layouts', origin: 'ours' },
        ],
      },
    ],
  },
  {
    id: 'anatomy',
    name: 'Anatomy harness',
    origin: 'upstream',
    note: 'the authors’ code, run as published',
    children: [
      { id: 'anatomy_e1', name: 'E1 masked recall', origin: 'upstream' },
      { id: 'anatomy_e4', name: 'E4 distractor haystack', origin: 'upstream' },
      { id: 'anatomy_e5', name: 'E5 colliding keys', origin: 'upstream' },
    ],
  },
  {
    id: 'tinystories',
    name: 'TinyStories',
    origin: 'external',
    note: 'public language data, read from a local copy',
  },
];

function Branch({ nodes }: { nodes: TreeNode[] }) {
  return (
    <ul className="ds-tree__list">
      {nodes.map((node) => (
        <li key={node.id} className="ds-tree__item">
          <div className="ds-tree__node">
            <span className="ds-tree__name">{node.name}</span>
            <OriginBadge origin={node.origin} />
            {node.note && <span className="ds-muted">{node.note}</span>}
          </div>
          {node.children && node.children.length > 0 && <Branch nodes={node.children} />}
        </li>
      ))}
    </ul>
  );
}

/** Family tree of the generators, each labelled ported, ours, upstream or external. */
export function FamilyTree() {
  return (
    <figure className="ds-tree" aria-label="Family tree of the dataset generators">
      <div className="ds-tree__top">Eleven generators, four families</div>
      <Branch nodes={FAMILIES} />
      <WhatThisShows>
        each line runs from a source project to the generators built from it. The badge says where the code came from: ported (copied and
        checked against the original), ours (written here), upstream (the authors&apos; own code) or external (outside data).
      </WhatThisShows>
    </figure>
  );
}

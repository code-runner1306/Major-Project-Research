"""Training-data arms for the layout-fix study (protocol layout-fix-v1).

Each arm removes one cue that lets block-first overwrite examples be solved by position:
  A0_control          unchanged v4 data (byte-identical to recipe_data.build_data)
  A1_varied_distance  query distance cycles over 8..16 instead of always 8
  A2_no_block_first   v4 data with block-first layouts rejected (all target writes before every unrelated write)
  A3_mixed_rewrites   rewrite count cycles over 1..3 instead of always 1
Everything else (vocabulary, length, key count, two scored answers per example) is unchanged.
recipe_data.py is deliberately not modified: its hash is part of every existing checkpoint's source identity.
"""
from collections import Counter

import torch

from config import RetrievalConfig
from recipe_data import RecipeData, manifest
from retrieval import RetrievalDataset

ARMS = {
    'A0_control': dict(query_distances=(8,), overwrite_counts=(1,), reject_block_first=False),
    'A1_varied_distance': dict(query_distances=tuple(range(8, 17)), overwrite_counts=(1,), reject_block_first=False),
    'A2_no_block_first': dict(query_distances=(8,), overwrite_counts=(1,), reject_block_first=True),
    'A3_mixed_rewrites': dict(query_distances=(8,), overwrite_counts=(1, 2, 3), reject_block_first=False),
}
KEYS = 4


def is_block_first(record):
    return record.metadata['post_target_write_count'] == record.metadata['key_count'] - 1


def build_arm_data(arm, examples, seed, split='train', vocabulary=8192, length=64):
    spec = ARMS[arm]
    config = RetrievalConfig(vocabulary_size=vocabulary, sequence_lengths=(length + 1,), key_counts=(KEYS,),
                             query_distances=spec['query_distances'], overwrite_counts=spec['overwrite_counts'],
                             generator_version='mqar-overwrite-v4')
    limit = examples * 4
    dataset = RetrievalDataset(task='overwrite', examples=limit, config=config, seed=seed, split=split)
    records, index, rejected = [], 0, 0
    while len(records) < examples:
        if index >= limit:
            raise RuntimeError(f'{arm}: could not draw {examples} examples')
        record = dataset.example(index)
        index += 1
        if spec['reject_block_first'] and is_block_first(record):
            rejected += 1
            continue
        records.append(record)
    tokens = torch.stack([x.tokens for x in records])
    labels = torch.stack([x.labels for x in records])
    kinds = torch.full_like(labels, -1)
    stale = torch.full_like(labels, -1)
    erase = torch.zeros_like(tokens, dtype=torch.bool)
    for i, record in enumerate(records):
        if len(record.answers) != 2:
            raise ValueError('Every arm must keep exactly two scored answers per example')
        for answer in record.answers:
            kinds[i, answer.label_position] = 1 if answer.kind == 'latest' else 2
            if answer.kind == 'latest':
                stale[i, answer.label_position] = answer.stale_values[-1]
            if answer.label_position + 1 < tokens.size(1):
                erase[i, answer.label_position + 1] = True
    identity = manifest(tokens, labels, seed=seed, split=split)
    if arm == 'A0_control':
        # Same fields as recipe_data.build_data so the control hash is directly comparable.
        identity.update(generator='audited-v4-overwrite-extension', upstream_revision=None,
                        source_manifest=dataset.manifest(), emitted_sequence_length=length + 1, model_input_length=length)
        identity['source_manifest']['examples'] = examples
    else:
        identity.update(generator='layout-fix-v1', arm=arm, upstream_revision=None, source_manifest=dataset.manifest(),
                        emitted_sequence_length=length + 1, model_input_length=length)
    identity['layout_statistics'] = dict(
        draws=index, rejected_block_first=rejected,
        block_first=sum(map(is_block_first, records)),
        query_distance=dict(Counter(r.metadata['query_distance'] for r in records)),
        overwrite_count=dict(Counter(r.metadata['overwrite_count'] for r in records)),
        post_target_writes=dict(Counter(r.metadata['post_target_write_count'] for r in records)),
        scored_answers=int((labels != -100).sum()), write_tokens=sum(2 * r.metadata['memory_load'] for r in records))
    return RecipeData(tokens, labels, kinds, stale, erase, identity)

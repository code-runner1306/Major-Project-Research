"""KATA-style repeated-key overwrite data (protocol kata-overwrite-v1).

Follows the published task description of Ghriss and Chakraborty (KATA, arXiv 2607.17419, Section 5.2): each of K
unique keys is written twice, each time with a different value, and the model must return the most recent binding.
Their token format and write ordering are not given in the public text, so these choices are ours and must be
reported as a reimplementation from the description:

  * Zoology conventions: keys from [2, V/2), values from [V/2, V), filler token 0, no answer tokens in the input.
  * Training K cycles over 4, 6, 8 (KATA trains K in {4..32} at length up to 256; our length is 64).
  * The 2K writes are a uniformly shuffled stream, so the two writes of a key are interleaved with other writes.
  * After the memory, every key is queried once in random order as [key, 0]. The label sits at the query key
    position (answer-free, as in Zoology). Remaining positions are filler 0.

In training, "second write of a key" and "latest write of a key" always coincide. Evaluation cells
(make_eval_cell) break that by writing a target key once, three or four times.
"""
import hashlib
import random

import torch

from recipe_data import RecipeData, manifest

ARMS = {'kata_twice': dict(keys=(4, 6, 8), writes_per_key=2)}
FILLER = 0


def _rng(seed, split, index):
    digest = hashlib.sha256(f'kata-overwrite-v1:{seed}:{split}:{index}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def example(rng, keys, writes, length, vocabulary):
    """writes: list with the number of writes for each key. Returns tokens, labels, per-query write values."""
    key_tokens = rng.sample(range(2, vocabulary // 2), keys)
    stream = [k for k in range(keys) for _ in range(writes[k])]
    rng.shuffle(stream)
    values = rng.sample(range(vocabulary // 2, vocabulary), len(stream))
    tokens, history = [], {k: [] for k in range(keys)}
    for k, value in zip(stream, values):
        tokens += [key_tokens[k], value]
        history[k].append(value)
    labels = [-100] * len(tokens)
    queries = list(range(keys))
    rng.shuffle(queries)
    query_meta = []
    for k in queries:
        query_meta.append(dict(key=k, position=len(tokens), writes=list(history[k])))
        tokens += [key_tokens[k], FILLER]
        labels += [history[k][-1], -100]
    if len(tokens) > length:
        raise ValueError(f'{keys} keys with {sum(writes)} writes do not fit length {length}')
    pad = length - len(tokens)
    return tokens + [FILLER] * pad, labels + [-100] * pad, query_meta


def _to_data(rows, identity):
    tokens = torch.tensor([r[0] for r in rows])
    labels = torch.tensor([r[1] for r in rows])
    kinds = torch.where(labels != -100, 1, -1)
    stale = torch.full_like(labels, -1)
    for i, (_, _, meta) in enumerate(rows):
        for q in meta:
            if len(q['writes']) > 1:
                stale[i, q['position']] = q['writes'][-2]
    erase = torch.zeros_like(tokens, dtype=torch.bool)
    return RecipeData(tokens, labels, kinds, stale, erase, manifest(tokens, labels, **identity))


def build_kata_data(examples, seed, split='train', vocabulary=8192, length=64):
    key_choices = [k for k in ARMS['kata_twice']['keys'] if 6 * k <= length]
    rows = []
    for i in range(examples):
        keys = key_choices[i % len(key_choices)]
        rows.append(example(_rng(seed, split, i), keys, [2] * keys, length, vocabulary))
    data = _to_data(rows, dict(seed=seed, split=split, generator='kata-overwrite-v1', keys=key_choices,
                                writes_per_key=2))
    data.identity['query_meta'] = None
    return data


EVAL_CELLS = {
    # name: (key counts cycled, writes for the target key, writes for every other key)
    'in_distribution': ((4, 6, 8), 2, 2),
    'target_once': ((4, 6, 8), 1, 2),
    'target_three': ((4, 6, 8), 3, 2),
    'target_four': ((4, 6, 8), 4, 2),
    'all_once': ((4, 6, 8), 1, 1),
    'all_three': ((4, 6), 3, 3),
}


def make_eval_cell(name, examples, seed, vocabulary=8192, length=64):
    """Rows with per-query metadata; key 0 is the target (its query position is randomised by the shuffle)."""
    key_counts, target_writes, other_writes = EVAL_CELLS[name]
    rows = []
    for i in range(examples):
        keys = key_counts[i % len(key_counts)]
        writes = [target_writes] + [other_writes] * (keys - 1)
        rows.append(example(_rng(seed, f'eval:{name}', i), keys, writes, length, vocabulary))
    data = _to_data(rows, dict(seed=seed, split=f'eval:{name}', generator='kata-overwrite-v1'))
    return data, [r[2] for r in rows]


def oracle_audit(data, vocabulary=8192):
    """Independent check: every label is the value after the key's most recent memory write."""
    failures = 0
    for tokens, labels in zip(data.tokens.tolist(), data.labels.tolist()):
        for position, label in enumerate(labels):
            if label == -100:
                continue
            key = tokens[position]
            latest = None
            for j in range(position - 1):
                if tokens[j] == key and tokens[j + 1] >= vocabulary // 2:
                    latest = tokens[j + 1]
            failures += latest != label
    return dict(passed=failures == 0, failures=failures, oracle='most recent key->value write before the query')

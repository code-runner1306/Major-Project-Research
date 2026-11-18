"""Repeated-key overwrite data for replicating KATA (arXiv 2607.17419), Section 5.2 and Appendix F.

Specified by the paper:
  * "Each of K unique keys is written twice with two different values, and the model must retrieve the most recent
    binding." Training mixes K in {4, 8, 16, 32} at sequence lengths up to 256; evaluation K in {64, 128, 256, 384}
    with T <= 3072 (Section 5.2).
  * Zoology MQAR conventions (F.1, F.3): vocabulary 8192 split into key and value halves; bindings first, then the K
    keys reissued in random order, each followed by an answer slot; loss only at answer positions.

Not specified, and the authors' data code is not public (upstream Zoology has no overwrite generator). Our choice is
the most direct extension of the F.1 MQAR layout:
  * Write pass 1: all K keys once, in random order, each with a fresh value. Write pass 2: all K keys again, in a new
    random order, each with a new value. Then the K queries in random order as [key, 0], 0 being the answer slot.
    T = 6K, so K = 32 gives 192 <= 256 and K = 384 gives 2304 <= 3072, consistent with the stated lengths.
  * Training uses 100,000 examples (the Zoology multiquery_ar default), split evenly over the four K values.
  * The answer is scored at the query key position, which is the same prediction as filling the answer slot.

Evaluation-only variants change the number of write passes (1, 3 or 4) or add a third pass for half of the keys,
which separates "latest write" from "second write" and from "the write in pass 2".
"""
import numpy as np
import torch

VOCAB = 8192
TRAIN_KEYS = (4, 8, 16, 32)
OOD_KEYS = (64, 128, 256, 384)
SPLITS = {'train': 1, 'validation': 2, 'early_stop': 3, 'evaluation': 4}


def generate(examples, keys, passes, seed, split, partial_last=False, vocabulary=VOCAB):
    """Returns tokens (n, T), labels (n, T) and writes (n, K, passes) with -1 for absent writes, plus query keys order.

    partial_last: the final pass rewrites only a random half of the keys (others keep their earlier latest value).
    """
    rng = np.random.default_rng([seed, SPLITS[split], keys, passes, int(partial_last)])
    length = 2 * keys * passes + 2 * keys
    tokens = np.zeros((examples, length), dtype=np.int64)
    labels = np.full((examples, length), -100, dtype=np.int64)
    writes = np.full((examples, keys, passes), -1, dtype=np.int64)
    key_space, value_space = np.arange(1, vocabulary // 2), np.arange(vocabulary // 2, vocabulary)
    for row in range(examples):
        key_tokens = rng.choice(key_space, keys, replace=False)
        values = rng.choice(value_space, keys * passes, replace=False).reshape(passes, keys)
        cursor = 0
        for p in range(passes):
            order = rng.permutation(keys)
            if partial_last and p == passes - 1:
                order = order[:keys // 2]
            for k in order:
                tokens[row, cursor], tokens[row, cursor + 1] = key_tokens[k], values[p, k]
                writes[row, k, p] = values[p, k]
                cursor += 2
        for k in rng.permutation(keys):
            latest = writes[row, k][writes[row, k] >= 0][-1]
            tokens[row, cursor] = key_tokens[k]          # answer slot at cursor + 1 stays 0
            labels[row, cursor] = latest
            cursor += 2
    if partial_last:  # drop the unused tail left by the shorter last pass
        used = length - keys
        tokens, labels = tokens[:, :used], labels[:, :used]
        # queries were written after the partial pass, so shift is already accounted for by the cursor
    return torch.tensor(tokens), torch.tensor(labels), torch.tensor(writes)


def training_set(examples=100000, seed=42):
    per_k = examples // len(TRAIN_KEYS)
    return {k: generate(per_k, k, 2, seed, 'train')[:2] for k in TRAIN_KEYS}


def oracle_check(tokens, labels, vocabulary=VOCAB):
    """Independent check: each label equals the value after the most recent write of the queried key."""
    failures = 0
    for row_tokens, row_labels in zip(tokens.tolist(), labels.tolist()):
        latest = {}
        for position, label in enumerate(row_labels):
            token = row_tokens[position]
            if label != -100:
                failures += latest.get(token) != label
            elif position + 1 < len(row_tokens) and row_tokens[position + 1] >= vocabulary // 2 and token < vocabulary // 2 and token != 0:
                latest[token] = row_tokens[position + 1]
    return failures

"""Read-only counterexamples and shortcut baselines for the methodology audit."""
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from config import RetrievalConfig
from recurrence import reference_recurrence
from retrieval import RetrievalDataset

root = Path(__file__).resolve().parents[1]
task = json.loads((root / 'output/granularity_exp_1/task_configs/overwrite_10M_C1-scalar-default_seed_42_train.json').read_text(encoding='utf-8-sig'))
config = RetrievalConfig(**task['retrieval_config'])

# Published GDN: decay the state before evaluating the delta residual.
one = torch.ones(1, 1, 1, 1)
_, implemented = reference_recurrence(one, one, one, one * .5, one * .5, torch.ones(1, 1, 1, 1))
print('scalar_recurrence', {'implemented': implemented.item(), 'published': .75})

# With beta=0, isolate the axis of channel decay in value-by-key storage.
q = torch.tensor([[[[0., 1.]]]])
k = torch.tensor([[[[1., 0.]]]])
initial = torch.tensor([[[[1., 2.], [3., 4.]]]])
alpha = torch.tensor([[[[.2, .8]]]])
_, implemented = reference_recurrence(q, k, torch.zeros_like(k), alpha, torch.zeros(1, 1, 1, 1), initial)
expected = initial * alpha[:, 0].unsqueeze(-2)
print('channel_decay_axis', {'implemented': implemented.tolist(), 'published_in_same_storage': expected.tolist()})

for family in ('overwrite', 'mqar'):
    data = RetrievalDataset(task=family, examples=512, config=config, seed=200043, split='held_out')
    correct = total = 0
    for index in range(len(data)):
        ex = data.example(index)
        first_query = ex.answers[0].query_position
        # Read value tokens only, strictly before the first query; no queried keys.
        values = [int(v) for v in ex.tokens[:first_query] if int(v) >= config.vocabulary_size // 2]
        for ordinal, answer in enumerate(ex.answers):
            predicted = values[-2 + ordinal] if family == 'overwrite' else (values[-1] if ordinal == 0 else -1)
            correct += predicted == answer.target
            total += 1
    print('key_free_baseline', family, {'correct': correct, 'total': total, 'accuracy': correct / total})
    short = RetrievalDataset(task=family, examples=1, config=replace(config, sequence_lengths=(128,)), seed=200043, split='held_out').example(0)
    long = RetrievalDataset(task=family, examples=1, config=replace(config, sequence_lengths=(512,)), seed=200043, split='held_out').example(0)
    stop = max(a.label_position for a in short.answers) + 1
    print('length_control', family, {'identical_scored_prefix': torch.equal(short.tokens[:stop], long.tokens[:stop]), 'last_scored_position': stop - 1})

for path in sorted((root / 'output/granularity_exp_1/runs').glob('*/*/*/seed_42/metrics.json')):
    data = json.loads(path.read_text())
    grouping = data.get('posthoc_gate_grouping', {}).get('metrics_by_group_count', {})
    print('record', path.relative_to(root).as_posix(), {'parameters': data['parameter']['parameter_count'], 'gate_sharing_accuracy': {k: v.get('exact_query_accuracy') for k, v in grouping.items()}})
print('current_recurrence_source_sha256', hashlib.sha256((root / 'recurrence.py').read_bytes()).hexdigest())
lf = (root / 'recurrence.py').read_bytes().replace(b'\r\n', b'\n')
print('VM_CRLF_recurrence_source_sha256', hashlib.sha256(lf.replace(b'\n', b'\r\n')).hexdigest())

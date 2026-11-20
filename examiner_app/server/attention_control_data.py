"""Attention control page: saved results, a live status check, and one real example from the diagnostic's generator.

Read-only. Every number comes from the saved files named in content_briefs/attention-control.md (read at request time).
Nothing is trained or written. The example uses the same generator settings as attention_control_diagnostic.datasets.
"""
import json
import re
import sys
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from checkpoints import repo_relative  # noqa: E402

ATTN = ROOT / 'research_results' / 'attention_control_20261115'
KAGGLE = ROOT / 'research_results' / 'kaggle_seed42_20261114'
DECAY_NOTE = ROOT / 'decay_design_results.md'
OUTPUT = ROOT / 'output'

FAMILIES = ('mqar', 'overwrite')
KAGGLE_CONDITIONS = {'C1': ('scalar', 'C1-scalar-default'), 'C3': ('channel-wise', 'C3-channel-default')}
ANSWER_FIELDS = ('exact_query_accuracy', 'first_query_accuracy', 'later_query_accuracy',
                 'latest_value_accuracy', 'unrelated_association_retention', 'stale_value_error_rate')


class AttentionControlError(Exception):
    def __init__(self, status, detail):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def _counts(block):
    """Exact counts with their accuracy, from a summary's answer_free block."""
    return dict(correct=block['correct'], total=block['total'], accuracy=block['exact_query_accuracy'])


def _same_protocol_run(family, arch):
    run_dir = ATTN / family / family / arch
    summary = _json(run_dir / 'summary.json')
    settings = _json(run_dir / 'settings.json')
    audit = _json(ATTN / family / f'{family}_audit.json')
    answer = summary['confirmation']['answer_free']
    history = _json(run_dir / 'validation_history.json')
    return dict(
        answer_free=_counts(answer),
        answer_free_detail={k: answer[k] for k in ANSWER_FIELDS if k != 'exact_query_accuracy'},
        development_curve=[dict(step=row['step'], accuracy=row['development_answer_accuracy']) for row in history],
        parameters=settings['model_parameters'],
        clipped_step_fraction=summary['clipped_step_fraction'],
        steps=summary['steps'],
        wall_seconds=round(summary['wall_seconds']),
        status=summary['status'],
        key_free_baselines=_key_free_best(audit),
    )


def _key_free_best(audit):
    rows = audit['confirmation']['baselines']
    best = max((v for k, v in rows.items() if k != 'keyed_last_write_oracle'), key=lambda v: v['accuracy'])
    return dict(name=max((k for k in rows if k != 'keyed_last_write_oracle'),
                         key=lambda k: rows[k]['accuracy']), accuracy=best['accuracy'], correct=best['correct'],
                total=best['total'])


def _kaggle_run(family, condition):
    _, cid = KAGGLE_CONDITIONS[condition]
    metrics = _json(KAGGLE / condition / 'runs' / family / cid / 'seed_42' / 'metrics.json')
    retrieval = metrics['retrieval']
    return dict(
        answer_free=dict(correct=retrieval['correct'], total=retrieval['total'],
                         accuracy=retrieval['exact_query_accuracy']),
        final_validation_loss=metrics['final_validation']['loss'],
        parameters=metrics['parameter']['trainable_parameter_count'],
    )


@lru_cache(maxsize=1)
def _decay_gaps():
    """Channel-wise minus scalar validation loss from the paper table (section 4), mean column."""
    out = {}
    for line in DECAY_NOTE.read_text(encoding='utf-8').splitlines():
        cells = [c.strip() for c in line.split('|')]
        if len(cells) >= 6 and re.match(r'^\d+M TinyStories$', cells[1]) and 'KDA' in cells[2]:
            out[cells[1].split()[0]] = float(cells[4])
    return out


@lru_cache(maxsize=1)
def _backbone():
    settings = _json(ATTN / 'overwrite' / 'overwrite' / 'gdn' / 'settings.json')
    return settings['backbone']


def _checkpoints():
    """Attention checkpoints saved anywhere in the repo. Kaggle checkpoints are not retained, so this is normally empty."""
    found = []
    for root in (OUTPUT, ATTN):
        if root.exists():
            found += [p for p in root.rglob('latest_checkpoint.pt') if 'attention' in p.as_posix().lower()]
    return sorted(repo_relative(p) for p in found)


@lru_cache(maxsize=1)
def status():
    found = _checkpoints()
    if found:
        return dict(live=True, checkpoints=found, reason='Attention checkpoints found; the live try-it is not built yet.')
    return dict(live=False, checkpoints=[],
                reason='No attention checkpoint is saved in this repository, so only saved results are shown.')


def summary():
    backbone = _backbone()
    width_heads = backbone['n_head'] * backbone['d_head']
    same = {family: {arch: _same_protocol_run(family, arch) for arch in ('gdn', 'attention')} for family in FAMILIES}
    kaggle = {family: {cond: _kaggle_run(family, cond) for cond in KAGGLE_CONDITIONS} for family in FAMILIES}
    overwrite = same['overwrite']
    return dict(
        status=status(),
        protocol=dict(
            same=dict(steps=2000, learning_rate=0.001, training_examples=10000, held_out_examples=512,
                      sequence_length=128, keys=4, query_distance=8, overwrite_count=1, vocabulary=8192),
            kaggle=dict(steps=1000, learning_rate=0.001, held_out_examples=512, context_length=512, seed=42),
        ),
        same_protocol=same,
        scalar_vs_channel=kaggle,
        paper_context=dict(
            channel_minus_scalar_gap={'10M': _decay_gaps().get('10M'), '100M': _decay_gaps().get('100M')},
            note='Literature result (TinyStories language modelling), not one of the tests above.',
        ),
        memory=dict(
            recurrent_state_numbers_per_layer=width_heads * backbone['d_head'],
            attention_cache_numbers_per_token_per_layer=2 * width_heads,
            heads=backbone['n_head'], head_width=backbone['d_head'], layers=backbone['n_layer'],
            width=backbone['d_model'],
        ),
        headline=dict(
            value=f"{overwrite['attention']['answer_free']['accuracy'] * 100:.0f}% vs "
                  f"{overwrite['gdn']['answer_free']['accuracy'] * 100:.0f}%",
            label='overwrite test, held-out answers correct: attention vs scalar decay',
        ),
        provenance=dict(files=[
            repo_relative(ATTN / 'mqar' / 'mqar_audit.json'), repo_relative(ATTN / 'overwrite' / 'overwrite_audit.json'),
            repo_relative(ATTN / 'overwrite' / 'overwrite' / 'attention' / 'summary.json'),
            repo_relative(ATTN / 'overwrite' / 'overwrite' / 'gdn' / 'summary.json'),
            repo_relative(ATTN / 'mqar' / 'mqar' / 'attention' / 'summary.json'),
            repo_relative(ATTN / 'mqar' / 'mqar' / 'gdn' / 'summary.json'),
            repo_relative(KAGGLE / 'C1' / 'runs' / 'mqar' / 'C1-scalar-default' / 'seed_42' / 'metrics.json'),
            repo_relative(KAGGLE / 'C3' / 'runs' / 'mqar' / 'C3-channel-default' / 'seed_42' / 'metrics.json'),
            repo_relative(DECAY_NOTE), repo_relative(ROOT / 'ATTENTION_DIAGNOSTIC.md'),
        ]),
    )


# One overwrite example from the diagnostic's generator (held-out split, seed + 300000), with roles for every position
# that the metadata names. Only the overwrite family is shown: its MQAR metadata does not list every key/value pair.
DIAG_CONFIG = dict(vocabulary_size=8192, train_examples=10000, validation_examples=128, held_out_examples=512,
                   sequence_lengths=(128,), key_counts=(4,), query_distances=(8,), overwrite_counts=(1,),
                   generator_version='mqar-overwrite-v4')


def example(index: int, seed: int = 42):
    import retrieval
    from config import RetrievalConfig

    config = RetrievalConfig(**DIAG_CONFIG)
    dataset = retrieval.RetrievalDataset(task='overwrite', examples=512, config=config, seed=seed + 300000,
                                         split='held_out')
    ex = dataset.example(index)
    tokens = [int(t) for t in ex.tokens.tolist()]
    meta = ex.metadata
    roles = ['filler'] * len(tokens)
    target_values = set(meta['write_positions'])
    for p in meta['all_write_positions']:
        if p < len(tokens):
            roles[p] = 'target_value' if p in target_values else 'value'
            roles[p - 1] = 'target_key' if p in target_values else 'key'
    for p in meta['query_positions']:
        if p < len(tokens):
            roles[p] = 'marker'
    queries = []
    for a in ex.answers:
        if a.label_position < len(tokens):
            roles[a.label_position] = 'query_latest' if a.kind == 'latest' else 'query_unrelated'
        queries.append(dict(kind=a.kind, query_position=a.query_position, key_position=a.label_position,
                            answer=int(a.target), latest_value_position=a.latest_value_position))
    return dict(
        index=index, seed=seed, sequence_length=meta['sequence_length'], token_count=len(tokens),
        tokens=tokens, roles=roles, queries=queries,
        writes=dict(target=meta['write_positions'], all=meta['all_write_positions'],
                    key_positions=meta['write_key_positions']),
    )

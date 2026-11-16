from evaluate_published_checkpoints import build_confirmation
from recipe_data import build_data


def test_overwrite_confirmation_uses_fresh_held_out_domain():
    cfg = dict(seed=42, vocabulary=8192, length=64)
    actual = build_confirmation('overwrite', cfg)
    expected = build_data('overwrite',512,300042,'held_out',8192,64)
    assert actual.identity['sha256'] == expected.identity['sha256']
    assert actual.identity['seed'] == 300042
    assert actual.identity['split'] == 'held_out'


def test_mqar_confirmation_examples_unchanged():
    cfg = dict(seed=42, vocabulary=8192, length=64)
    actual = build_confirmation('mqar', cfg)
    previous = build_data('mqar',512,300042,'confirmation',8192,64)
    assert actual.identity == previous.identity

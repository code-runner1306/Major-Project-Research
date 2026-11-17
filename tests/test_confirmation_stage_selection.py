import json
import pytest
from confirm_recipe_study import select_runs
from run_recipe_study import identity,PROTOCOL


def make_matrix(root,epochs=16,stage='explore'):
    for initialization in ['default','calibrated']:
        for variant in ['scalar','channel-wise']:
            folder=root/initialization/variant;folder.mkdir(parents=True)
            cfg=dict(protocol=PROTOCOL,stage=stage,family='mqar',seed=42,initialization=initialization,
                     variant=variant,epochs=epochs,source_hashes=identity())
            (folder/'summary.json').write_text(json.dumps(dict(status='complete',settings=cfg,accepted_optimizer_steps=196*epochs)))
            (folder/'latest_checkpoint.pt').write_bytes(b'preflight-only-test')


def test_explore_requires_explicit_selection(tmp_path):
    make_matrix(tmp_path)
    assert len(select_runs(tmp_path,['mqar'],[42],[],'explore'))==4
    with pytest.raises(ValueError,match='Incomplete study matrix'):
        select_runs(tmp_path,['mqar'],[42],[])


def test_study_default_still_accepts_complete_64_epoch_matrix(tmp_path):
    make_matrix(tmp_path,64,'study')
    assert len(select_runs(tmp_path,['mqar'],[42],[]))==4


def test_missing_checkpoint_fails_before_evaluation(tmp_path):
    make_matrix(tmp_path)
    (tmp_path/'default/scalar/latest_checkpoint.pt').unlink()
    with pytest.raises(ValueError,match='Missing checkpoint'):
        select_runs(tmp_path,['mqar'],[42],[],'explore')


def test_overwrite_manifest_json_roundtrip_preserves_identity():
    from recipe_data import build_data
    from confirm_recipe_study import same_manifest
    original=build_data('overwrite',2,100042,'train',8192,64).identity
    saved=json.loads(json.dumps(original))
    assert same_manifest(original,saved)
    saved['sha256']='different content'
    assert not same_manifest(original,saved)

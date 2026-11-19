"""Paired MQAR filler-extension evaluation of the existing 16-epoch checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from artifacts import atomic_write_json
from confirm_recipe_study import select_runs, same_manifest
from recipe_data import RecipeData, audit_mqar, manifest, mqar
from recipe_evaluation import evaluate
from recipe_model import RecipeLM

ROOT = Path('output/calibrated_recipe_study')
OUT = Path('research_results/mqar_length_extension_20261119')
LENGTHS = (64, 256, 1024)
SEEDS = (42, 43, 44)
EXAMPLES = 256
OFFSET = 1950000
PREFIX = 8


def run(output: Path = OUT):
    torch.set_num_threads(2)
    device = torch.device('cpu')
    selected = select_runs(ROOT, ['mqar'], SEEDS, [], 'explore')
    output.mkdir(parents=True, exist_ok=False)
    atomic_write_json(output/'protocol.json', {
        'description': 'Post hoc frozen-checkpoint evaluation. Insert filler zeros after the four key/value writes.',
        'lengths': LENGTHS, 'training_seeds': SEEDS, 'generator_seed_offset': OFFSET,
        'examples_per_seed': EXAMPLES, 'device': str(device), 'torch': str(torch.__version__),
        'initializations': ['default','calibrated'], 'batch': 8,
        'metric': 'exact answer-token accuracy', 'checkpoint_endpoint': '16 epochs / 3136 optimizer steps',
    })
    rows = []
    for seed in SEEDS:
        tokens, targets = mqar(EXAMPLES, length=64, seed=OFFSET+seed)
        train_tokens, train_targets = mqar(100000, length=64, seed=100000+seed)
        training = set(zip(map(tuple, train_tokens.tolist()), map(tuple, train_targets.tolist())))
        assert not any(pair in training for pair in zip(map(tuple, tokens.tolist()), map(tuple, targets.tolist())))
        del training, train_tokens, train_targets
        cases = {}
        for length in LENGTHS:
            extra = length-64
            extended = torch.cat((tokens[:, :PREFIX], tokens.new_zeros((EXAMPLES,extra)), tokens[:, PREFIX:]), dim=1)
            labels = torch.cat((targets[:, :PREFIX], targets.new_full((EXAMPLES,extra),-100), targets[:, PREFIX:]), dim=1)
            assert audit_mqar(extended,labels)['passed']
            cases[length] = RecipeData(extended, labels, torch.where(labels!=-100,0,-1),
                torch.full_like(labels,-1), torch.zeros_like(extended,dtype=torch.bool),
                manifest(extended,labels,seed=seed+OFFSET,injected_filler=extra))
        for (_, checkpoint_seed, init, variant), (summary, record) in selected.items():
            if checkpoint_seed!=seed:
                continue
            path = summary.parent/'latest_checkpoint.pt'
            with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
                checkpoint = torch.load(path,map_location=device,weights_only=True)
            if not same_manifest(checkpoint['settings'],record['settings']):
                raise ValueError(f'Checkpoint metadata mismatch: {path}')
            model = RecipeLM(variant,8192,64,128,'chunked',seed)
            model.load_state_dict(checkpoint['model'],strict=True)
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            for length,data in cases.items():
                metrics=evaluate(model,data,device,batch=8)
                rows.append(dict(seed=seed,initialization=init,variant=variant,length=length,
                    checkpoint=str(path),checkpoint_sha256=digest,**metrics))
                print(seed,init,variant,length,metrics['exact_query_accuracy'],flush=True)
            atomic_write_json(output/'results.json',{'status':'running','results':rows})
    atomic_write_json(output/'results.json',{'status':'complete','results':rows})


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT, help='New directory for protocol and results.')
    run(parser.parse_args().output)

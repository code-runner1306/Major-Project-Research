"""Frozen confirmation evaluation of the four final published-recipe models."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import zipfile

import torch
from artifacts import atomic_write_json
from recipe_data import build_data, split_overlap, audit_mqar
from recipe_evaluation import evaluate
from recipe_model import RecipeLM


def checkpoint_path(folder):
    target = folder / 'latest_checkpoint.pt'
    if target.exists():
        return target
    extracted = folder / 'latest_checkpoint'
    if extracted.is_dir() and not (extracted / 'data.pkl').is_file():
        candidates = list(extracted.rglob('data.pkl'))
        if len(candidates) == 1:
            extracted = candidates[0].parent
    if not (extracted / 'data.pkl').is_file() or not (extracted / 'version').is_file():
        raise FileNotFoundError(f'Missing checkpoint or extracted PyTorch archive in {folder}')
    temporary = folder / 'rebuilding_checkpoint.tmp'
    try:
        with zipfile.ZipFile(temporary, 'x', compression=zipfile.ZIP_STORED) as archive:
            for path in sorted(extracted.rglob('*')):
                if path.is_file():
                    archive.write(path, 'latest_checkpoint/' + path.relative_to(extracted).as_posix())
        temporary.rename(target)
    finally:
        if temporary.exists():
            temporary.unlink()
    return target


def source_check(settings):
    # Windows checkout newline conversion changes bytes without changing code.
    for name, expected in settings['source_hashes'].items():
        raw = Path(name).read_bytes().replace(b'\r\n', b'\n')
        hashes = {hashlib.sha256(raw).hexdigest(), hashlib.sha256(raw.replace(b'\n', b'\r\n')).hexdigest()}
        if expected not in hashes:
            raise ValueError(f'Checkpoint source mismatch: {name}')


def load_checkpoint(path, family):
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    cfg = checkpoint['settings']
    source_check(cfg)
    if cfg['family'] != family or checkpoint['completed_steps'] != 12544 or checkpoint['completed_epochs'] != 64:
        raise ValueError(f'Expected final epoch64/12544-step {family} checkpoint: {path}')
    return checkpoint


def build_confirmation(family, cfg):
    # V4 retrieval uses explicit RNG domains; confirmation is a protocol role,
    # implemented with its supported held_out domain and a fresh fixed seed.
    split = 'held_out' if family == 'overwrite' else 'confirmation'
    return build_data(family,512,cfg['seed']+300000,split,cfg['vocabulary'],cfg['length'])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint-root', type=Path, default=Path('.'))
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    p.add_argument('--prepare-only', action='store_true')
    args = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = {}
    for family in ['mqar','overwrite']:
        for variant in ['scalar','channel-wise']:
            folder = args.checkpoint_root / ('published_' + family) / variant
            if not folder.exists() and variant == 'channel-wise':
                folder = folder.with_name('channel_wise')
            paths[family,variant] = checkpoint_path(folder)
    if args.prepare_only:
        for (family,variant), path in paths.items():
            checkpoint = load_checkpoint(path, family)
            cfg = checkpoint['settings']
            model = RecipeLM(variant,cfg['vocabulary'],cfg['length'],cfg['width'],cfg['backend'],cfg['seed'])
            model.load_state_dict(checkpoint['model'], strict=True)
            del model, checkpoint
            print(f'Validated {family}/{variant}: {path} ({path.stat().st_size} bytes)', flush=True)
        return
    args.output.mkdir(parents=True, exist_ok=False)
    # Freeze before generating examples or observing any confirmation score.
    protocol = dict(examples=512, seed_offset=300000, batch=32, primary='answer_free',
                    generator_split={'mqar':'confirmation','overwrite':'held_out'},
                    selection='all four final epoch64 checkpoints; no best-checkpoint selection',
                    scope='same-distribution confirmation, not harder-task or multi-seed evidence')
    atomic_write_json(args.output / 'confirmation_protocol.json', protocol)
    device = torch.device('cuda:0' if args.device=='auto' and torch.cuda.is_available()
                          else 'cpu' if args.device=='auto' else args.device)
    report = dict(status='running', device=str(device), torch=torch.__version__, results={})
    started = time.monotonic()
    try:
        for family in ['mqar','overwrite']:
            paired = None
            data = None
            for variant in ['scalar','channel-wise']:
                path = paths[family,variant]
                ckpt = load_checkpoint(path, family)
                cfg = ckpt['settings']
                if paired is not None and cfg != paired:
                    raise ValueError('Paired checkpoint settings differ')
                paired = cfg
                if data is None:
                    data = build_confirmation(family, cfg)
                    overlaps = {}
                    for split,offset,count in [('train',100000,cfg['train_examples']),
                                               ('validation',200000,cfg['validation_examples'])]:
                        other = build_data(family,count,cfg['seed']+offset,split,cfg['vocabulary'],cfg['length'])
                        overlaps[split] = split_overlap(other,data)
                        del other
                    if any(overlaps.values()):
                        raise ValueError('Confirmation overlaps previous splits')
                    if family == 'mqar' and not audit_mqar(data.tokens,data.labels)['passed']:
                        raise ValueError('MQAR confirmation label audit failed')
                    atomic_write_json(args.output / f'{family}_data_manifest.json',
                                      dict(identity=data.identity, content_overlap=overlaps))
                model = RecipeLM(variant,cfg['vocabulary'],cfg['length'],cfg['width'],cfg['backend'],cfg['seed'])
                model.load_state_dict(ckpt['model'], strict=True)
                del ckpt
                model.to(device)
                tick = time.monotonic()
                primary = evaluate(model,data,device,batch=32)
                secondary = (evaluate(model,data,device,batch=32,history='teacher_forced')
                             if family=='overwrite' else dict(primary,history_conditioning='teacher_forced'))
                result = dict(answer_free=primary,teacher_forced=secondary,
                              evaluation_seconds=time.monotonic()-tick,
                              checkpoint=str(path),checkpoint_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
                report['results'][family+'/'+variant] = result
                atomic_write_json(args.output / 'confirmation_results.json', report)
                print(f'{family}/{variant}: {primary["correct"]}/{primary["total"]} '
                      f'({primary["exact_query_accuracy"]:.4%})', flush=True)
                del model
                if device.type=='cuda':
                    torch.cuda.empty_cache()
        report['status'] = 'complete'
    except BaseException as error:
        report.update(status='failed',error=str(error))
        raise
    finally:
        report['wall_seconds'] = time.monotonic()-started
        atomic_write_json(args.output / 'confirmation_results.json', report)


if __name__ == '__main__':
    main()

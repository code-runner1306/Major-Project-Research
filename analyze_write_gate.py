"""Write-gate, key-consistency and readout diagnostics for the first-replacement error.

No training. Loads existing 16-epoch overwrite checkpoints, re-runs the GDN mixers
with the exact serial delta-rule recurrence, and records per-layer quantities at
every write of the queried (target) key:

* beta    -- write strength sigmoid(b_proj(x)) at the key and value token of each write
* keys    -- cosine of each write's key with the latest write's key and with the
             query vector used at the answer position (short-conv "key drift")
* readout -- exact attribution of the memory readout S_T q_T at the answer position:
             S_T q_T = sum_t w_t v_t with w_t = beta_t k_t^T A_{t+1}..A_T q_T and
             A_t = diag(alpha_t)(I - beta_t k_t k_t^T); reported at each target write

It then applies causal patches inside the recurrence (beta and key edits restricted
to the target key's writes, plus an unrelated-write control) and re-scores answers.
Discovery evidence only: stress data use seed offset 500000 (already observed in
memory_stress_tests_20261117); confirmation offset 900000 stays untouched.
"""
import argparse
import csv
import json
import time
from pathlib import Path

import torch
from torch.nn import functional as F

from artifacts import atomic_write_json
from config import RetrievalConfig
from confirm_recipe_study import select_runs
from recipe_model import RecipeGDN, RecipeLM, published_l2norm
from retrieval import RetrievalDataset
from stress_recipe_checkpoints import make_overwrite

CELLS = {1: 'trained_one_rewrite', 2: 'two_rewrites', 3: 'three_rewrites', 4: 'four_rewrites'}
PATCH_CELL = 3
OUTCOMES = ('correct', 'initial', 'first_rewrite', 'second_rewrite', 'third_rewrite', 'other')


# ----------------------------------------------------------------------------- data

class Positions:
    """Per-example token positions taken from the generator metadata."""

    def __init__(self, records):
        self.key = [r.metadata['write_key_positions'] for r in records]
        self.value = [r.metadata['write_positions'] for r in records]
        self.values = [r.metadata['write_values'] for r in records]
        self.latest_label = [next(a.label_position for a in r.answers if a.kind == 'latest') for r in records]
        self.unrelated_label = [next(a.label_position for a in r.answers if a.kind == 'unrelated') for r in records]
        # Control: the unrelated write immediately after the latest target write (always exists in v4).
        self.all_value = [r.metadata['all_write_positions'] for r in records]
        self.control_value = [next(p for p in r.metadata['all_write_positions'] if p > r.metadata['write_positions'][-1])
                              for r in records]

    def batch(self, start, stop):
        out = Positions.__new__(Positions)
        for name in ('key', 'value', 'values', 'latest_label', 'unrelated_label', 'control_value', 'all_value'):
            setattr(out, name, getattr(self, name)[start:stop])
        return out


def build_cell(examples, seed, rewrites):
    case = make_overwrite(examples, seed + 500000, 4, 8, rewrites, seed + 100000)
    cfg = RetrievalConfig(vocabulary_size=8192, sequence_lengths=(65,), key_counts=(4,), query_distances=(8,),
                          overwrite_counts=(rewrites,), generator_version='mqar-overwrite-v4')
    dataset = RetrievalDataset(task='overwrite', examples=examples, config=cfg, seed=seed + 500000, split='held_out')
    records = [dataset.example(i) for i in range(examples)]
    if not torch.equal(torch.stack([r.tokens for r in records]), case.data.tokens):
        raise AssertionError('Metadata records differ from the stress case tokens')
    for r in records:
        if len(r.metadata['write_positions']) != rewrites + 1:
            raise AssertionError('Unexpected number of target writes')
    tokens = case.data.tokens.clone()
    tokens[case.data.erase_answers] = 1  # answer-free contract of recipe_evaluation.evaluate
    return dict(case=case, tokens=tokens, positions=Positions(records), rewrites=rewrites)


# ------------------------------------------------------------------ instrumented mixer

class Context:
    def __init__(self):
        self.patch, self.layers, self.positions, self.record = 'none', set(), None, None


def serial_with_readout(q, k, v, alpha, beta):
    """Same arithmetic as recipe_model.serial_rectangular, returning per-step outputs."""
    state = q.new_zeros(q.size(0), q.size(2), v.size(-1), k.size(-1))
    outputs = []
    for index in range(q.size(1)):
        state = state * alpha[:, index].unsqueeze(-2)
        key = k[:, index].unsqueeze(-1)
        residual = v[:, index].unsqueeze(-1) - state @ key
        state = state + beta[:, index].unsqueeze(-1) * residual @ key.transpose(-1, -2)
        outputs.append((state @ q[:, index].unsqueeze(-1)).squeeze(-1))
    return torch.stack(outputs, dim=1)


def apply_patch(ctx, layer, k, beta):
    if ctx.patch == 'none' or layer not in ctx.layers:
        return k, beta
    k, beta, p = k.clone(), beta.clone(), ctx.positions
    for i in range(k.size(0)):
        keys, values = p.key[i], p.value[i]
        pairs = list(zip(keys, values))
        if ctx.patch == 'beta_one_latest':
            for pos in pairs[-1]:
                beta[i, pos] = 1.
        elif ctx.patch == 'beta_copy_rewrite1':
            for kp, vp in pairs[2:]:
                beta[i, kp], beta[i, vp] = beta[i, pairs[1][0]], beta[i, pairs[1][1]]
        elif ctx.patch == 'beta_zero_rewrite1':
            for pos in pairs[1]:
                beta[i, pos] = 0.
        elif ctx.patch == 'key_align_latest':
            for kp, vp in pairs[:-1]:
                k[i, kp], k[i, vp] = k[i, pairs[-1][0]], k[i, pairs[-1][1]]
        elif ctx.patch == 'control_beta_one_unrelated':
            vp = p.control_value[i]
            beta[i, vp - 1], beta[i, vp] = 1., 1.
        else:
            raise ValueError(f'Unknown patch {ctx.patch}')
    return k, beta


def instrumented_forward(module, x, *, ctx, layer):
    def projected(projection, convolution, width):
        raw = projection(x).transpose(1, 2)
        return F.silu(convolution(F.pad(raw, (3, 0))).transpose(1, 2)).reshape(x.size(0), x.size(1), module.heads, width)
    q = published_l2norm(projected(module.q_proj, module.q_conv, module.head_k))
    k = published_l2norm(projected(module.k_proj, module.k_conv, module.head_k))
    v = projected(module.v_proj, module.v_conv, module.head_v)
    alpha = module.decay(x)
    beta = module.b_proj(x).view(x.size(0), x.size(1), module.heads, 1).sigmoid()
    k, beta = apply_patch(ctx, layer, k, beta)
    output = serial_with_readout(q * module.head_k ** -.5, k, v, alpha, beta)
    if ctx.record is not None:
        ctx.record.append({name: tensor.detach().float().cpu() for name, tensor in dict(
            q=q[:, :, 0], k=k[:, :, 0], v=v[:, :, 0], beta=beta[:, :, 0, 0], alpha=alpha[:, :, 0],
            readout=output[:, :, 0]).items()})
    normalized = output.float() * torch.rsqrt(output.float().square().mean(-1, keepdim=True) + 1e-5)
    gated = normalized * module.o_norm * F.silu(module.g_proj(x).view_as(output))
    return module.o_proj(gated.flatten(-2))


def instrument(model):
    ctx = Context()
    layers = [m for m in model.modules() if isinstance(m, RecipeGDN)]
    if len(layers) != 4:
        raise AssertionError('Expected four GDN layers')
    for index, layer in enumerate(layers):
        layer.forward = lambda x, _m=layer, _i=index: instrumented_forward(_m, x, ctx=ctx, layer=_i)
    return ctx


def restore(model):
    for layer in model.modules():
        if isinstance(layer, RecipeGDN) and 'forward' in layer.__dict__:
            del layer.forward


# --------------------------------------------------------------------------- scoring

@torch.no_grad()
def predict(model, cell, ctx, batch, record=False):
    predictions, records = [], []
    device = next(model.parameters()).device
    tokens, labels = cell['tokens'].to(device), cell['case'].data.labels.to(device)
    for start in range(0, len(tokens), batch):
        stop = start + batch
        ctx.positions = cell['positions'].batch(start, stop)
        ctx.record = [] if record else None
        mask = labels[start:stop] != -100
        predictions.append(model.scored_logits(tokens[start:stop], mask).argmax(-1).cpu())
        if record:
            records.append(ctx.record)
    ctx.record = None
    return torch.cat(predictions), records


def classify(predictions, cell):
    """Outcome of the latest-target query and correctness of the unrelated query, per example."""
    data, p = cell['case'].data, cell['positions']
    mask = data.labels != -100
    per_row = mask.sum(1).tolist()
    split = torch.split(predictions, per_row)
    outcomes, unrelated = [], []
    for i, row in enumerate(split):
        positions = mask[i].nonzero().flatten().tolist()
        lookup = dict(zip(positions, row.tolist()))
        answer, values = lookup[p.latest_label[i]], p.values[i]
        if answer == values[-1]:
            outcome = 'correct'
        elif answer in values[:-1]:
            outcome = OUTCOMES[1 + values[:-1].index(answer)]
        else:
            outcome = 'other'
        outcomes.append(outcome)
        unrelated.append(lookup[p.unrelated_label[i]] == int(data.labels[i, p.unrelated_label[i]]))
    return outcomes, unrelated


def score(outcomes, unrelated):
    n = len(outcomes)
    counts = {o: outcomes.count(o) for o in OUTCOMES}
    return dict(examples=n, latest_value_accuracy=counts['correct'] / n,
                unrelated_retention=sum(unrelated) / n,
                first_rewrite_error_rate=counts['first_rewrite'] / n, outcome_counts=counts)


# ----------------------------------------------------------------------- diagnostics

def cos(a, b):
    return F.cosine_similarity(a, b, dim=-1)


def attribution(q, k, alpha, beta, stop):
    """Scalar weight of each earlier value token in the readout at position `stop`."""
    weights = q.new_zeros(stop + 1)
    u = q[stop]
    for t in range(stop, -1, -1):
        weights[t] = beta[t] * (k[t] @ u)
        u = alpha[t] * (u - beta[t] * k[t] * (k[t] @ u))
    return weights


def diagnostics(records, cell, outcomes):
    """Per layer, per write occurrence, split by outcome of the latest-target query."""
    rewrites = cell['rewrites']
    rows = []
    offset = 0
    for batch_records in records:
        size = batch_records[0]['beta'].size(0)
        p = cell['positions'].batch(offset, offset + size)
        for layer, r in enumerate(batch_records):
            for i in range(size):
                outcome = outcomes[offset + i]
                kp, vp, lq = p.key[i], p.value[i], p.latest_label[i]
                weights = attribution(r['q'][i] * r['q'].size(-1) ** -.5, r['k'][i], r['alpha'][i], r['beta'][i], lq)
                readout = r['readout'][i, lq]
                rebuilt = (weights.unsqueeze(-1) * r['v'][i, :lq + 1]).sum(0)
                if not torch.allclose(rebuilt, readout, rtol=1e-3, atol=1e-4):
                    raise AssertionError('Readout attribution does not reconstruct the recurrence output')
                total = weights.abs().sum().clamp_min(1e-12)
                for j in range(rewrites + 1):
                    rows.append(dict(
                        layer=layer, occurrence=j, outcome=outcome,
                        beta_key=float(r['beta'][i, kp[j]]), beta_value=float(r['beta'][i, vp[j]]),
                        alpha_value=float(r['alpha'][i, vp[j]].mean()),
                        key_cos_latest_keypos=float(cos(r['k'][i, kp[j]], r['k'][i, kp[-1]])),
                        key_cos_latest_valuepos=float(cos(r['k'][i, vp[j]], r['k'][i, vp[-1]])),
                        key_cos_query_keypos=float(cos(r['k'][i, kp[j]], r['q'][i, lq])),
                        key_cos_query_valuepos=float(cos(r['k'][i, vp[j]], r['q'][i, lq])),
                        readout_cos_value=float(cos(readout, r['v'][i, vp[j]])),
                        weight_key=float(weights[kp[j]]), weight_value=float(weights[vp[j]]),
                        share_key=float(weights[kp[j]].abs() / total), share_value=float(weights[vp[j]].abs() / total)))
                cp = p.control_value[i]
                rows.append(dict(layer=layer, occurrence='unrelated_control', outcome=outcome,
                                 beta_key=float(r['beta'][i, cp - 1]), beta_value=float(r['beta'][i, cp]),
                                 alpha_value=float(r['alpha'][i, cp].mean()),
                                 weight_key=float(weights[cp - 1]), weight_value=float(weights[cp]),
                                 share_key=float(weights[cp - 1].abs() / total),
                                 share_value=float(weights[cp].abs() / total)))
                # Where the rest of the readout comes from: the answer-query tokens themselves.
                target = set(kp) | set(vp)
                other = {x for v_pos in p.all_value[i] for x in (v_pos - 1, v_pos)} - target
                query = {lq - 1, lq}
                rest = set(range(lq + 1)) - target - other - query
                for role, members in (('role_target_writes', target), ('role_other_writes', other),
                                      ('role_query_tokens', query), ('role_filler_and_other_query', rest)):
                    rows.append(dict(layer=layer, occurrence=role, outcome=outcome,
                                     share_value=float(weights[sorted(members)].abs().sum() / total) if members else 0.))
        offset += size
    return rows


def summarize(rows):
    groups = {}
    for row in rows:
        for split in (row['outcome'], 'all'):
            groups.setdefault((row['layer'], str(row['occurrence']), split), []).append(row)
    summary = []
    for (layer, occurrence, outcome), members in sorted(groups.items()):
        entry = dict(layer=layer, occurrence=occurrence, outcome=outcome, n=len(members))
        for name in members[0]:
            values = [m[name] for m in members if isinstance(m.get(name), float)]
            if values:
                entry[name] = sum(values) / len(values)
        summary.append(entry)
    return summary


# ----------------------------------------------------------------------------- main

PATCHES = [('none', 'all'), ('beta_one_latest', 'all'), ('beta_copy_rewrite1', 'all'),
           ('beta_zero_rewrite1', 'all'), ('key_align_latest', 'all'), ('control_beta_one_unrelated', 'all')]
PATCHES += [(name, layer) for name in ('beta_one_latest', 'beta_zero_rewrite1', 'key_align_latest') for layer in range(4)]


def select_joint(root, seeds, run_ids):
    from run_joint_recipe_study import identity
    joint = {}
    for path in root.rglob('summary.json'):
        summary = json.loads(path.read_text())
        cfg = summary.get('settings', {})
        if (cfg.get('protocol') != 'joint-timescale-sensitivity-v1' or cfg.get('family') != 'overwrite'
                or cfg.get('seed') not in seeds or cfg.get('stage') != 'explore' or cfg.get('initialization') != 'joint'):
            continue
        if run_ids and not any(x in path.parts for x in run_ids):
            continue
        key = ('overwrite', cfg['seed'], 'joint', cfg['variant'])
        if key in joint:
            raise ValueError('Duplicate joint condition; use --joint-run-id')
        if (summary.get('status') != 'complete' or cfg.get('epochs') != 16
                or summary.get('accepted_optimizer_steps') != 3136 or cfg.get('source_hashes') != identity()):
            raise ValueError(f'Incomplete or incompatible joint checkpoint: {path}')
        if not (path.parent / 'latest_checkpoint.pt').is_file():
            raise ValueError(f'Missing checkpoint: {path.parent}')
        joint[key] = (path, summary)
    expected = {('overwrite', s, 'joint', v) for s in seeds for v in ('scalar', 'channel-wise')}
    if set(joint) != expected:
        raise ValueError(f'Incomplete joint matrix: {sorted(expected - set(joint))}')
    return joint


def load(path, variant, seed):
    model = RecipeLM(variant, 8192, 64, 128, 'chunked', seed)
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        saved = torch.load(path.parent / 'latest_checkpoint.pt', map_location='cpu', weights_only=True)
    model.load_state_dict(saved['model'], strict=True)
    return model.eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/calibrated_recipe_study/matched-decay-v1'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 43, 44])
    parser.add_argument('--initializations', nargs='+', default=['default', 'calibrated'])
    parser.add_argument('--variants', nargs='+', default=['scalar', 'channel-wise'])
    parser.add_argument('--examples', type=int, default=256)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--joint-root', type=Path, help='Also analyse the joint-initialization overwrite models')
    parser.add_argument('--joint-run-id', nargs='*', default=[])
    args = parser.parse_args()
    device = torch.device('cuda' if args.device == 'cuda' or (args.device == 'auto' and torch.cuda.is_available()) else 'cpu')
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA requested but unavailable')
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    selected = select_runs(args.root, ['overwrite'], args.seeds, [], 'explore')
    initializations = list(args.initializations)
    if args.joint_root:
        selected.update(select_joint(args.joint_root, args.seeds, args.joint_run_id))
        initializations.append('joint')
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    protocol = dict(purpose=__doc__, examples=args.examples, seed_offset=500000, keys=4, distance=8,
                    cells=CELLS, patch_cell=PATCH_CELL, patches=PATCHES, conditioning='answer_free', device=str(device),
                    torch=torch.__version__, gpu=torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                    recurrence='serial reference (exact delta rule), checked against stock chunked predictions',
                    checkpoints={str(k): str(v[0].parent) for k, v in selected.items()})
    atomic_write_json(args.output / 'protocol.json', protocol)
    started = time.monotonic()
    results, diagnostic_rows = [], []
    for seed in args.seeds:
        cells = {r: build_cell(args.examples, seed, r) for r in CELLS}
        for init in initializations:
            for variant in args.variants:
                path, _ = selected['overwrite', seed, init, variant]
                model = load(path, variant, seed).to(device)
                tag = dict(seed=seed, initialization=init, variant=variant)
                for rewrites, cell in cells.items():
                    mask = cell['case'].data.labels != -100
                    with torch.no_grad():
                        stock = torch.cat([model.scored_logits(cell['tokens'][i:i + args.batch].to(device),
                                                               mask[i:i + args.batch].to(device)).argmax(-1).cpu()
                                           for i in range(0, len(mask), args.batch)])
                    ctx = instrument(model)
                    predictions, records = predict(model, cell, ctx, args.batch, record=True)
                    agreement = float((predictions == stock).float().mean())
                    if agreement < .995:
                        raise AssertionError(f'Serial/chunked disagreement {agreement:.4f} for {tag}')
                    outcomes, unrelated = classify(predictions, cell)
                    results.append(dict(**tag, cell=CELLS[rewrites], rewrites=rewrites, patch='none', patch_layers='all',
                                        serial_chunked_agreement=agreement, **score(outcomes, unrelated)))
                    for row in diagnostics(records, cell, outcomes):
                        diagnostic_rows.append(dict(**tag, rewrites=rewrites, **row))
                    if rewrites == PATCH_CELL:
                        for patch, layers in PATCHES[1:]:
                            ctx.patch, ctx.layers = patch, set(range(4)) if layers == 'all' else {layers}
                            patched, _ = predict(model, cell, ctx, args.batch)
                            o, u = classify(patched, cell)
                            results.append(dict(**tag, cell=CELLS[rewrites], rewrites=rewrites, patch=patch,
                                                patch_layers=str(layers), **score(o, u)))
                        ctx.patch, ctx.layers = 'none', set()
                    restore(model)
                print(f'{seed}/{init}/{variant}: done ({time.monotonic() - started:.0f}s)', flush=True)
                atomic_write_json(args.output / 'results.json', dict(status='running', results=results))
    summary = {}
    for key in {(r['initialization'], r['variant'], r['rewrites']) for r in diagnostic_rows}:
        rows = [r for r in diagnostic_rows if (r['initialization'], r['variant'], r['rewrites']) == key]
        summary['/'.join(map(str, key))] = summarize(rows)
    pooled = {str(r): summarize([x for x in diagnostic_rows if x['rewrites'] == r]) for r in CELLS}
    atomic_write_json(args.output / 'results.json', dict(status='complete', wall_seconds=time.monotonic() - started,
                                                         results=results))
    atomic_write_json(args.output / 'diagnostic_summary.json', dict(by_condition=summary, pooled=pooled))
    with (args.output / 'patch_results.csv').open('w', newline='', encoding='utf-8') as file:
        fields = ['seed', 'initialization', 'variant', 'cell', 'rewrites', 'patch', 'patch_layers',
                  'latest_value_accuracy', 'unrelated_retention', 'first_rewrite_error_rate']
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    with (args.output / 'diagnostics_per_write.csv').open('w', newline='', encoding='utf-8') as file:
        fields = sorted({k for r in diagnostic_rows for k in r})
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(diagnostic_rows)
    print(f'Complete in {time.monotonic() - started:.0f}s', flush=True)


if __name__ == '__main__':
    main()

"""PROTOTYPE (throwaway): export real checkpoint evidence for the examiner UI prototype.

Runs the saved layout-fix checkpoints (A0-A4, scalar + channel-wise, seeds 42-44) on brand-new
overwrite data (seed offset 2_100_000, unused by any earlier experiment), traces one delta-rule
update inside a real model, and gathers saved training curves / paper results into one JSON file
the web prototype reads. No training, no checkpoint is modified.

    .venv/Scripts/python.exe prototype_examiner_ui/export_data.py [--examples 64] [--skip-overlap]
"""
import argparse
import glob
import json
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import confirm_block_layout as cbl  # noqa: E402
from analyze_write_gate import load  # noqa: E402
from layout_fix_data import ARMS, build_arm_data  # noqa: E402
from recipe_model import published_l2norm, serial_rectangular  # noqa: E402
import torch.nn.functional as F  # noqa: E402

FRESH_OFFSET = 2_100_000
cbl.SEED_OFFSET = FRESH_OFFSET
LAYOUT_ROOT = ROOT / 'output/layout_fix_study/layout-fix-v1'
OUT = Path(__file__).resolve().parent / 'web/public/data.json'
SEEDS = (42, 43, 44)
VARIANTS = ('scalar', 'channel-wise')
CELLS = [('main_r1', 1), ('main_r3', 3)]          # trained layout vs extra rewrites
LAYOUTS = ('block_first', 'general_nonblock')


def r(x, n=4):
    return round(float(x), n)


def checkpoint(arm, seed, variant):
    hits = glob.glob(str(LAYOUT_ROOT / '*' / arm / f'seed_{seed}' / variant / 'latest_checkpoint.pt'))
    if len(hits) != 1:
        raise FileNotFoundError(f'{arm}/{seed}/{variant}: {hits}')
    return Path(hits[0])


def fresh_cells(seed):
    out = {}
    for index, (name, rewrites) in enumerate(CELLS):
        chosen, info = cbl.sample_layouts(seed, index, 4, 8, rewrites, list(LAYOUTS), ARGS.examples)
        out[name] = dict(rewrites=rewrites, info=info,
                         layouts={lay: cbl.to_cell(recs, rewrites, dict(seed=seed, cell=name, layout=lay))
                                  for lay, recs in chosen.items()})
    return out


def row_key(tokens, labels):
    return hash((tuple(tokens.tolist()), tuple(labels.tolist())))


def overlap_audit(seed, cells):
    """Exact (tokens, labels) overlap of every fresh example with every arm's 100k training pool."""
    fresh = {}
    for cell in cells.values():
        for lay in cell['layouts'].values():
            data = lay['case'].data
            for t, l in zip(data.tokens, data.labels):
                fresh[row_key(t, l)] = True
    result = {}
    for arm in ARMS:
        train = build_arm_data(arm, 100000, seed + 100000)
        hits = sum(row_key(t, l) in fresh for t, l in zip(train.tokens, train.labels))
        result[arm] = hits
    return dict(fresh_examples=len(fresh), overlapping_by_arm=result)


def which_write(prediction, values):
    names = ['initial', 'first_rewrite', 'second_rewrite', 'third_rewrite']
    for i, v in enumerate(values):
        if prediction == v:
            return 'latest' if i == len(values) - 1 else names[i]
    return 'other'


def score(model, lay):
    outcomes, unrelated, latest_answers = cbl.evaluate(model, lay, 'cpu', 64)
    values = lay['positions'].values
    kinds = [which_write(p, v) for p, v in zip(latest_answers, values)]
    n = len(kinds)
    unrel_hits = []
    mask = lay['case'].data.labels != -100
    return dict(latest=r(sum(k == 'latest' for k in kinds) / n),
                answer_source={k: kinds.count(k) for k in sorted(set(kinds))},
                examples=n, latest_answers=latest_answers, kinds=kinds)


def describe_example(lay, i, predictions_by_model):
    rec = lay['records'][i]
    md = rec.metadata
    toks = lay['tokens'][i].tolist()
    roles = ['filler'] * len(toks)
    for p in md['all_write_positions']:
        if p < len(toks):
            roles[p] = 'value'
            roles[p - 1] = 'key'
    for p in md['write_positions']:
        if p < len(toks):
            roles[p] = 'target_value'
            roles[p - 1] = 'target_key'
    for a in rec.answers:
        roles[a.label_position] = 'query_latest' if a.kind == 'latest' else 'query_unrelated'
        if a.label_position - 1 >= 0 and toks[a.label_position - 1] == 0:
            roles[a.label_position - 1] = 'marker'
    return dict(tokens=toks, roles=roles, pattern=cbl.pattern_of(rec), write_values=md['write_values'],
                latest_label_position=lay['positions'].latest_label[i],
                rule_answers=cbl.rule_answers(rec), predictions=predictions_by_model)


# ---------------------------------------------------------------- delta-rule trace inside a real model
def mixer_inputs(model, tokens):
    captured = {}
    hooks = [blk.sequence_mixer.register_forward_pre_hook(
        lambda m, a, idx=idx: captured.__setitem__(idx, a[0].detach()))
        for idx, blk in enumerate(model.network.backbone.layers)]
    with torch.no_grad():
        logits = model.network(tokens)
    for h in hooks:
        h.remove()
    return captured, logits


def mixer_tensors(mixer, x):
    def projected(projection, convolution, width):
        raw = projection(x).transpose(1, 2)
        return F.silu(convolution(F.pad(raw, (3, 0))).transpose(1, 2)).reshape(x.size(0), x.size(1), mixer.heads, width)
    q = published_l2norm(projected(mixer.q_proj, mixer.q_conv, mixer.head_k)) * mixer.head_k ** -.5
    k = published_l2norm(projected(mixer.k_proj, mixer.k_conv, mixer.head_k))
    v = projected(mixer.v_proj, mixer.v_conv, mixer.head_v)
    alpha = mixer.decay(x)
    beta = mixer.b_proj(x).view(x.size(0), x.size(1), mixer.heads, 1).sigmoid()
    return q, k, v, alpha, beta


def trace_update(model, tokens, step, layer=3):
    """State before/after the delta-rule update at `step` in `layer`, recomputed by hand and checked."""
    captured, _ = mixer_inputs(model, tokens[None])
    mixer = model.network.backbone.layers[layer].sequence_mixer
    with torch.no_grad():
        x = captured[layer]
        q, k, v, alpha, beta = mixer_tensors(mixer, x)
        _, state_prev = serial_rectangular(q[:, :step], k[:, :step], v[:, :step], alpha[:, :step], beta[:, :step])
        S = state_prev[0, 0]                         # [Dv=256, Dk=128]
        a_t = alpha[0, step, 0]                      # [1] or [128]
        b_t = beta[0, step, 0, 0]
        k_t, v_t = k[0, step, 0], v[0, step, 0]
        S_decay = S * a_t.unsqueeze(0)              # decay each key column
        read = S_decay @ k_t                         # what memory currently returns for k_t  [256]
        residual = v_t - read
        S_new = S_decay + b_t * torch.outer(residual, k_t)
        _, oracle = serial_rectangular(q[:, :step + 1], k[:, :step + 1], v[:, :step + 1],
                                       alpha[:, :step + 1], beta[:, :step + 1])
        err = (S_new - oracle[0, 0]).abs().max().item()
        read_after = S_new @ k_t
    rows, cols = 32, 32                              # display crop; full shape reported separately
    return dict(
        layer=layer + 1, step=step, token=int(tokens[step]),
        shapes=dict(state=list(S.shape), k=list(k_t.shape), v=list(v_t.shape), alpha=list(a_t.shape),
                    beta=[1], q=list(q[0, step, 0].shape)),
        beta=r(b_t), alpha=[r(a, 5) for a in a_t.tolist()],
        k=[r(t) for t in k_t.tolist()], v_head=[r(t) for t in v_t[:rows].tolist()],
        read_before_head=[r(t) for t in read[:rows].tolist()],
        residual_norm=r(residual.norm()), v_norm=r(v_t.norm()),
        read_error_before=r((v_t - read).norm() / v_t.norm()),
        read_error_after=r((v_t - read_after).norm() / v_t.norm()),
        state_before=[[r(c, 3) for c in row[:cols].tolist()] for row in S[:rows]],
        state_decayed=[[r(c, 3) for c in row[:cols].tolist()] for row in S_decay[:rows]],
        state_after=[[r(c, 3) for c in row[:cols].tolist()] for row in S_new[:rows]],
        state_norm_before=r(S.norm()), state_norm_after=r(S_new.norm()),
        update_rank=int(torch.linalg.matrix_rank(S_new - S_decay).item()),
        max_abs_error_vs_model_recurrence=err,
    )


def gate_profile(model, tokens):
    captured, _ = mixer_inputs(model, tokens[None])
    layers = []
    with torch.no_grad():
        for idx, blk in enumerate(model.network.backbone.layers):
            mixer = blk.sequence_mixer
            alpha = mixer.decay(captured[idx])[0, :, 0]          # [T, gate_width]
            layers.append(dict(layer=idx + 1, alpha_shape=list(mixer.decay(captured[idx]).shape),
                               a_proj_weight=list(mixer.a_proj.weight.shape), A_log=list(mixer.A_log.shape),
                               dt_bias=list(mixer.dt_bias.shape),
                               alpha=[[r(a, 4) for a in row.tolist()] for row in alpha]))
    params = sum(p.numel() for p in model.parameters())
    return dict(parameters=params, layers=layers)


def curves():
    out = []
    for arm in ARMS:
        for seed in SEEDS:
            for variant in VARIANTS:
                path = checkpoint(arm, seed, variant).parent
                curve = json.loads((path / 'development_curve.json').read_text())
                summary = json.loads((path / 'summary.json').read_text())
                out.append(dict(arm=arm, seed=seed, variant=variant,
                                epochs=[c['epoch'] for c in curve],
                                train_loss=[r(c['train_loss'], 5) for c in curve],
                                dev_accuracy=[r(c['answer_free']['exact_query_accuracy']) for c in curve],
                                arm_dev_accuracy=[r(c['arm_answer_free']['exact_query_accuracy']) for c in curve],
                                final_dev=r(summary['final_development']['answer_free']['exact_query_accuracy'])))
    return out


def saved_a4_report():
    import csv
    rows = list(csv.DictReader(open(ROOT / 'research_results/layout_fix_a4_confirmation_20261118/cell_results.csv')))
    keep = {(c, l) for c, _ in CELLS for l in LAYOUTS}
    return [dict(arm=x['initialization'], variant=x['variant'], seed=int(x['seed']), cell=x['cell'],
                 layout=x['layout'], latest=r(x['latest_value_accuracy']), examples=int(x['examples']))
            for x in rows if (x['cell'], x['layout']) in keep]


def paper_results():
    g = json.loads((ROOT / 'third_party/dualgoose/results/granularity_report.json').read_text())
    return dict(
        table1=[dict(condition='Scalar, default', mqar=99.93, overwrite=99.32, latest=100.00, retention=98.63),
                dict(condition='Channel-wise, default', mqar=99.97, overwrite=98.80, latest=99.93, retention=97.66),
                dict(condition='Scalar, mean decay', mqar=99.90, overwrite=97.98, latest=100.00, retention=95.96),
                dict(condition='Channel-wise, mean decay', mqar=99.98, overwrite=97.36, latest=99.87, retention=94.86)],
        anatomy_cells=g['cells'], anatomy_verdicts=g['verdicts'])


def main():
    t0 = time.time()
    data = dict(meta=dict(fresh_seed_offset=FRESH_OFFSET, examples_per_layout=ARGS.examples,
                          generated=time.strftime('%Y-%m-%d %H:%M'), note='PROTOTYPE export; real checkpoints, CPU'))
    results, showcase, audits = [], {}, {}
    for seed in SEEDS:
        cells = fresh_cells(seed)
        audits[seed] = None if ARGS.skip_overlap else overlap_audit(seed, cells)
        print(f'seed {seed}: data ready, overlap {audits[seed]}  ({time.time() - t0:.0f}s)', flush=True)
        for arm in ARMS:
            for variant in VARIANTS:
                model = load(checkpoint(arm, seed, variant), variant, seed)
                for cname, cell in cells.items():
                    for lname, lay in cell['layouts'].items():
                        s = score(model, lay)
                        results.append(dict(arm=arm, variant=variant, seed=seed, cell=cname, layout=lname,
                                            latest=s['latest'], answer_source=s['answer_source'], examples=s['examples']))
                        if seed == 42 and cname == 'main_r3' and lname == 'block_first':
                            showcase.setdefault('preds', {})[f'{arm}/{variant}'] = s['latest_answers']
                            showcase.setdefault('kinds', {})[f'{arm}/{variant}'] = s['kinds']
        if seed == 42:
            showcase['cells'] = cells
        print(f'seed {seed}: models scored ({time.time() - t0:.0f}s)', flush=True)
    data['fresh_results'] = results
    data['overlap_audit'] = audits

    # Showcase example: first block-first r3 example where A0 scalar 42 is wrong and A4 scalar 42 is right.
    lay = showcase['cells']['main_r3']['layouts']['block_first']
    a0, a4 = showcase['kinds']['A0_control/scalar'], showcase['kinds']['A4_distance_and_rewrites/scalar']
    pick = next((i for i in range(len(a0)) if a0[i] != 'latest' and a4[i] == 'latest'), 0)
    preds = {k: v[pick] for k, v in showcase['preds'].items()}
    data['showcase'] = describe_example(lay, pick, preds)
    general = showcase['cells']['main_r1']['layouts']['general_nonblock']
    data['showcase_trained_layout'] = describe_example(general, 0, {})

    # Delta-rule trace at the latest target write (value token) of the showcase example.
    tokens = lay['tokens'][pick]
    latest_value_pos = lay['records'][pick].metadata['write_positions'][-1]
    data['delta_rule'] = {}
    data['gates'] = {}
    for variant in VARIANTS:
        for arm in ('A0_control', 'A4_distance_and_rewrites'):
            model = load(checkpoint(arm, 42, variant), variant, 42)
            data['delta_rule'][f'{arm}/{variant}'] = trace_update(model, tokens, latest_value_pos)
        data['gates'][variant] = gate_profile(load(checkpoint('A4_distance_and_rewrites', 42, variant), variant, 42), tokens)
    data['curves'] = curves()
    data['saved_a4_report'] = saved_a4_report()
    data['paper'] = paper_results()
    data['meta']['wall_seconds'] = round(time.time() - t0, 1)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(',', ':')))
    print(f'wrote {OUT} ({OUT.stat().st_size / 1e6:.2f} MB) in {time.time() - t0:.0f}s')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--examples', type=int, default=64)
    p.add_argument('--skip-overlap', action='store_true')
    ARGS = p.parse_args()
    torch.set_grad_enabled(False)
    main()

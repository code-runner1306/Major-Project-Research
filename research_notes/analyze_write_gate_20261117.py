"""Summarize research_results/write_gate_mechanism_20261117 into tables for REPORT.md."""
import json
import sys
from collections import defaultdict
from pathlib import Path
from statistics import mean

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else 'research_results/write_gate_mechanism_20261117')
results = json.loads((ROOT / 'results.json').read_text())
assert results['status'] == 'complete'
rows = results['results']
pooled = json.loads((ROOT / 'diagnostic_summary.json').read_text())['pooled']
out = []

# 1. Reconcile unpatched three-rewrite predictions with the earlier error breakdown.
earlier = json.loads(Path('research_results/memory_stress_tests_20261117/error_breakdown.json').read_text())['overwrite']
mismatches = 0
compared = 0
for e in earlier:
    r = next((x for x in rows if (x['seed'], x['initialization'], x['variant'], x['rewrites'], x['patch']) ==
             (e['seed'], e['initialization'], e['variant'], 3, 'none')), None)
    if r is None or r['examples'] != e['total_latest']:
        continue
    compared += 1
    c = r['outcome_counts']
    same = (c['correct'], c['initial'], c['first_rewrite'], c['second_rewrite']) == (
        e['correct_latest'], e['initial_value_errors'], e['first_rewrite_errors'], e['second_rewrite_errors'])
    mismatches += not same
out.append(f'Reconciliation with memory_stress_tests error_breakdown: {compared - mismatches}/{compared} compared models identical')
out.append(f"Min serial/chunked agreement: {min(r['serial_chunked_agreement'] for r in rows if 'serial_chunked_agreement' in r):.4f}")

# 2. Accuracy by rewrite count.
out.append('\n## Accuracy by rewrite count (mean over models)\n')
out.append('| Rewrites | Latest value | Unrelated retention | First-replacement answers | Errors that are first replacement |')
out.append('|---:|---:|---:|---:|---:|')
for k in (1, 2, 3, 4):
    sel = [r for r in rows if r['rewrites'] == k and r['patch'] == 'none']
    errors = sum(r['examples'] - r['outcome_counts']['correct'] for r in sel)
    first = sum(r['outcome_counts']['first_rewrite'] for r in sel)
    out.append(f"| {k} | {mean(r['latest_value_accuracy'] for r in sel):.2%} | {mean(r['unrelated_retention'] for r in sel):.2%} | "
               f"{mean(r['first_rewrite_error_rate'] for r in sel):.2%} | {first}/{errors} |")

# 3. Patch effects on the three-rewrite cell.
out.append('\n## Causal patches, three rewrites (mean change vs unpatched over models; models improved/worsened by >=2 points)\n')
out.append('| Patch | Layers | Latest value | Unrelated | First-replacement rate | Improved / worse |')
out.append('|---|---|---:|---:|---:|---|')
base = {(r['seed'], r['initialization'], r['variant']): r for r in rows if r['rewrites'] == 3 and r['patch'] == 'none'}
groups = defaultdict(list)
for r in rows:
    if r['rewrites'] == 3 and r['patch'] != 'none':
        groups[r['patch'], r['patch_layers']].append(r)
for (patch, layers), sel in groups.items():
    d = [(r['latest_value_accuracy'] - base[r['seed'], r['initialization'], r['variant']]['latest_value_accuracy'],
          r['unrelated_retention'] - base[r['seed'], r['initialization'], r['variant']]['unrelated_retention'],
          r['first_rewrite_error_rate'] - base[r['seed'], r['initialization'], r['variant']]['first_rewrite_error_rate'])
         for r in sel]
    up = sum(x[0] >= .02 for x in d); down = sum(x[0] <= -.02 for x in d)
    out.append(f"| {patch} | {layers} | {mean(x[0] for x in d):+.2%} | {mean(x[1] for x in d):+.2%} | {mean(x[2] for x in d):+.2%} | {up} / {down} |")

# 4. Pooled diagnostics, three rewrites.
def table(metric, title, occurrences=('0', '1', '2', '3', 'unrelated_control')):
    out.append(f'\n## {title} (three rewrites, pooled over models)\n')
    out.append('| Layer | Outcome | ' + ' | '.join(occurrences) + ' |')
    out.append('|---|---|' + '---:|' * len(occurrences))
    for layer in range(4):
        for outcome in ('correct', 'first_rewrite'):
            vals = []
            for occ in occurrences:
                e = next((x for x in pooled['3'] if x['layer'] == layer and x['occurrence'] == occ and x['outcome'] == outcome), None)
                vals.append(f"{e[metric]:.3f}" if e and metric in e else '-')
            out.append(f'| {layer} | {outcome} | ' + ' | '.join(vals) + ' |')

table('beta_value', 'Write strength beta at the value token of each target write (0 = initial, 3 = latest)')
table('beta_key', 'Write strength beta at the key token of each target write')
table('key_cos_query_keypos', 'Cosine(key at write key token, query at answer position)', ('0', '1', '2', '3'))
table('key_cos_query_valuepos', 'Cosine(key at write value token, query at answer position)', ('0', '1', '2', '3'))
table('share_key', 'Readout attribution share: key token of each target write', ('0', '1', '2', '3'))
table('share_value', 'Readout attribution share: value token of each target write', ('0', '1', '2', '3'))
table('share_value', 'Readout attribution share by token role',
      ('role_target_writes', 'role_other_writes', 'role_query_tokens', 'role_filler_and_other_query'))

# 5. Trained distribution comparison: beta at the two writes seen in training.
out.append('\n## Beta at value token, one rewrite (trained distribution), pooled\n')
out.append('| Layer | initial | rewrite |')
out.append('|---|---:|---:|')
for layer in range(4):
    v = [next(x for x in pooled['1'] if x['layer'] == layer and x['occurrence'] == o and x['outcome'] == 'all')['beta_value']
         for o in ('0', '1')]
    out.append(f'| {layer} | {v[0]:.3f} | {v[1]:.3f} |')

text = '\n'.join(out)
(ROOT / 'tables.md').write_text(text, encoding='utf-8')
print(text)

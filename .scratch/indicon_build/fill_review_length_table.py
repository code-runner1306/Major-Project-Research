"""Assemble the measured length-sweep cohort into the manuscript table."""
import json
from itertools import product
from pathlib import Path
from statistics import mean

result_dir = Path('research_results/mqar_length_extension_20261119')
sources = [result_dir/'results.json', Path('.scratch/indicon_build/mqar_seed44/results.json')]
checkpoints = {}
for source in sources:
    if source.is_file():
        for row in json.loads(source.read_text())['results']:
            key = row['seed'], row['initialization'], row['variant'], row['length']
            if key in checkpoints and checkpoints[key] != row:
                raise ValueError(f'Conflicting duplicate measurement: {key}')
            checkpoints[key] = row

initializations = ['default','calibrated']
variants = ['scalar','channel-wise']
lengths = [64,256,1024]
seeds = [42,43,44]
expected = set(product(seeds,initializations,variants,lengths))
if set(checkpoints) != expected:
    raise ValueError(f'{len(checkpoints)}/36 cells complete. Missing: {sorted(expected-set(checkpoints))}')
assert all(row['total']==1024 and row['evaluated_examples']==256 for row in checkpoints.values())
assert all(len({checkpoints[seed, init, arm, n]['data_sha256'] for init, arm in product(initializations,variants)})==1
           for seed,n in product(seeds,lengths))

rows = [checkpoints[k] for k in sorted(checkpoints)]
stats = {}
for init, arm, n in product(initializations, variants, lengths):
    stats[init,arm,n] = 100*mean(checkpoints[seed,init,arm,n]['exact_query_accuracy'] for seed in seeds)
table_rows = []
for init in initializations:
    display = 'Default' if init=='default' else 'Mean decay'
    for seed in seeds:
        pairs=[]
        for n in lengths:
            values=[100*checkpoints[seed,init,arm,n]['exact_query_accuracy'] for arm in variants]
            pairs.append(f'{values[0]:.2f}/{values[1]:.2f}')
        table_rows.append(f'{display}, {seed} & '+' & '.join(pairs)+' \\\\')
    pairs=[f'{stats[init,"scalar",n]:.2f}/{stats[init,"channel-wise",n]:.2f}' for n in lengths]
    table_rows.append(f'{display}, mean & '+' & '.join(pairs)+' \\\\')
    if init==initializations[0]:
        table_rows.append(r'\midrule')
at_long = [100*r['exact_query_accuracy'] for r in rows if r['length']==1024]
deltas = [stats[init,'channel-wise',1024]-stats[init,'scalar',1024] for init in initializations]
range_text = f'{min(at_long):.2f}\\% to {max(at_long):.2f}\\%'
default_fail=100*checkpoints[44,'default','channel-wise',1024]['exact_query_accuracy']
calibrated_fail=100*checkpoints[43,'calibrated','scalar',1024]['exact_query_accuracy']
summary = rf'''Table~\ref{{tab:length}} shows the results. At 1,024 tokens, the channel-wise minus scalar mean differences were {deltas[0]:+.2f} and {deltas[1]:+.2f} percentage points for default and mean-decay initialization, respectively. Across checkpoints, accuracy ranged from {range_text}. The default channel-wise model at seed 44 fell to {default_fail:.2f}\%, while the mean-decay scalar model at seed 43 fell to {calibrated_fail:.2f}\%. Both had near-perfect accuracy at 64 tokens. This test exposes strong failures in both gate types and a ranking that depends on initialization and seed. Filler can itself produce state updates, so the test cannot isolate the learned decay factors as the cause of these failures.

\begin{{table}}[t]
\caption{{Fixed-state MQAR filler extension. Each entry gives scalar/channel-wise exact-query accuracy (\%).}}
\label{{tab:length}}
\centering
\scriptsize
\setlength{{\tabcolsep}}{{4pt}}
\begin{{tabular}}{{lrrr}}
\toprule
Init., seed & 64 & 256 & 1,024 \\
\midrule
'''+'\n'.join(table_rows)+r'''
\bottomrule
\end{tabular}
\end{table}'''
p = Path('indicon_gate_paper/main.tex')
text = p.read_text(encoding='utf-8')
assert 'LENGTH_RESULTS_PLACEHOLDER' in text
p.write_text(text.replace('LENGTH_RESULTS_PLACEHOLDER',summary), encoding='utf-8')
(result_dir/'verified_results.json').write_text(json.dumps({'status':'complete','results':rows},indent=2)+'\n',encoding='utf-8')
print(summary)

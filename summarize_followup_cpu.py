"""Create follow-up tables/figures after both fixed-weight CPU studies finish."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def write_csv(path,rows):
    with path.open('w',newline='',encoding='utf-8') as file:
        w=csv.DictWriter(file,fieldnames=rows[0]);w.writeheader();w.writerows(rows)


def main():
    root=Path('research_results')
    intervention_root=root/'stronger_gate_tests_20261117';stress_root=root/'memory_stress_tests_20261117'
    interventions=json.loads((intervention_root/'results.json').read_text())
    stress=json.loads((stress_root/'results.json').read_text())
    assert interventions['status']=='complete' and len(interventions['models'])==12
    assert stress['status']=='complete' and len(stress['results'])==96
    model_rows=[]
    for m in interventions['models']:
        tests={r['test']:r['metrics']['exact_query_accuracy'] for r in m['tests']}
        permutations=[tests[f'global_shuffle_{seed}'] for seed in [90221,90222,90223]]
        model_rows.append(dict(family=m['family'],seed=m['seed'],initialization=m['initialization'],
            baseline=m['baseline']['exact_query_accuracy'],global_mean=tests['global_mean'],
            global_static=tests['global_static'],shuffle_mean=statistics.mean(permutations),
            shuffle_min=min(permutations),shuffle_max=max(permutations),
            **{f'layer_{layer+1}_mean_drop':m['baseline']['exact_query_accuracy']-tests[f'layer_{layer}_mean'] for layer in range(4)}))
    write_csv(intervention_root/'per_model_summary.csv',model_rows)
    group_rows=[]
    for family in ['mqar','overwrite']:
        for init in ['default','calibrated']:
            rows=[r for r in model_rows if (r['family'],r['initialization'])==(family,init)]
            group_rows.append(dict(family=family,initialization=init,
                **{metric:statistics.mean(r[metric] for r in rows) for metric in ['baseline','global_mean','global_static','shuffle_mean']}))
    write_csv(intervention_root/'condition_means.csv',group_rows)
    stress_rows=[]
    for family in ['mqar','overwrite']:
        cells=(['baseline','more_keys','longer_context','longer_more_keys'] if family=='mqar' else
               ['baseline','more_keys','farther_query','more_rewrites'])
        for init in ['default','calibrated']:
            for variant in ['scalar','channel-wise']:
                for cell in cells:
                    rows=[r['metrics'] for r in stress['results'] if (r['family'],r['initialization'],r['variant'],r['cell'])==
                          (family,init,variant,cell)]
                    assert len(rows)==3
                    stress_rows.append(dict(family=family,initialization=init,variant=variant,cell=cell,
                        accuracy_mean=statistics.mean(r['exact_query_accuracy'] for r in rows),
                        accuracy_sd=statistics.stdev(r['exact_query_accuracy'] for r in rows),
                        retention_mean=None if family=='mqar' else statistics.mean(r['unrelated_association_retention'] for r in rows),
                        latest_mean=None if family=='mqar' else statistics.mean(r['latest_value_accuracy'] for r in rows),
                        stale_mean=None if family=='mqar' else statistics.mean(r['stale_value_error_rate'] for r in rows)))
    write_csv(stress_root/'condition_means.csv',stress_rows)
    colors=['#284b63','#2a9d8f','#b36a32','#a23b72']
    fig,axes=plt.subplots(1,2,figsize=(12,4.5),sharey=True)
    for ax,family in zip(axes,['mqar','overwrite']):
        cells=(['baseline','more_keys','longer_context','longer_more_keys'] if family=='mqar' else
               ['baseline','more_keys','farther_query','more_rewrites'])
        for (init,v),color in zip([('default','scalar'),('default','channel-wise'),('calibrated','scalar'),('calibrated','channel-wise')],colors):
            means=[next(r for r in stress_rows if (r['family'],r['initialization'],r['variant'],r['cell'])==(family,init,v,c))['accuracy_mean']*100 for c in cells]
            ax.plot(range(4),means,'o-',color=color,label=f'{v}, {init}')
            for seed in [42,43,44]:
                values=[next(r for r in stress['results'] if (r['family'],r['seed'],r['initialization'],r['variant'],r['cell'])==(family,seed,init,v,c))['metrics']['exact_query_accuracy']*100 for c in cells]
                ax.scatter(range(4),values,color=color,alpha=.3,s=15)
        ax.set_xticks(range(4),[c.replace('_','\n') for c in cells]);ax.set_title(family.upper());ax.grid(alpha=.2);ax.set_ylim(0,102)
    axes[0].set_ylabel('Answer-free accuracy (%)');axes[1].legend(fontsize=8,frameon=False)
    fig.suptitle('Fixed checkpoints under memory pressure: means and individual seeds')
    fig.tight_layout();fig.savefig(stress_root/'stress_accuracy.png',dpi=180);plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(11,4),sharey=True)
    metrics=['baseline','global_mean','global_static','shuffle_mean']
    for ax,family in zip(axes,['mqar','overwrite']):
        for init,color in [('default','#284b63'),('calibrated','#a23b72')]:
            group=next(r for r in group_rows if (r['family'],r['initialization'])==(family,init))
            ax.plot(range(4),[group[k]*100 for k in metrics],'o-',color=color,label=init)
            for row in [r for r in model_rows if (r['family'],r['initialization'])==(family,init)]:
                ax.scatter(range(4),[row[k]*100 for k in metrics],color=color,alpha=.3,s=15)
        ax.set_xticks(range(4),['Original','Shared\nmean','Static per\nchannel','Shuffled\nchannels']);ax.set_title(family.upper());ax.set_ylim(0,102);ax.grid(alpha=.2)
    axes[0].set_ylabel('Training-side probe accuracy (%)');axes[1].legend(frameon=False)
    fig.suptitle('Which properties of channel gates do trained models rely on?')
    fig.tight_layout();fig.savefig(intervention_root/'global_interventions.png',dpi=180);plt.close(fig)

    matrix=np.array([[r[f'layer_{i}_mean_drop']*100 for i in range(1,5)] for r in model_rows])
    fig,ax=plt.subplots(figsize=(7,6))
    limit=max(1,float(abs(matrix).max()));im=ax.imshow(matrix,cmap='RdBu_r',vmin=-limit,vmax=limit,aspect='auto')
    ax.set_xticks(range(4),['Layer1','Layer2','Layer3','Layer4'])
    ax.set_yticks(range(12),[f'{r["family"]} {r["seed"]} {r["initialization"]}' for r in model_rows],fontsize=8)
    for row in range(12):
        for col in range(4):ax.text(col,row,f'{matrix[row,col]:.1f}',ha='center',va='center',fontsize=8)
    fig.colorbar(im,ax=ax,label='Accuracy drop (percentage points)')
    ax.set_title('Replace one layer by shared mean; leave others unchanged')
    fig.tight_layout();fig.savefig(intervention_root/'layer_interventions.png',dpi=180);plt.close(fig)

    from stress_recipe_checkpoints import make_mqar
    distances=[]
    for audit in stress['data_audits']:
        if audit['family']!='mqar':continue
        identity=audit['identity'];length=identity['shape'][1];keys=identity['keys']
        case=make_mqar(identity['shape'][0],identity['seed'],length,keys)
        assert case.data.identity==identity
        values=[]
        for tokens,labels in zip(case.data.tokens.tolist(),case.data.labels.tolist()):
            positions={tokens[2*i]:2*i+1 for i in range(keys)}
            values.extend(pos-positions[tokens[pos]] for pos,value in enumerate(labels) if value!=-100)
        distances.append(dict(seed=audit['seed'],cell=audit['cell'],mean=statistics.mean(values),
                              median=statistics.median(values),maximum=max(values)))
    write_csv(stress_root/'mqar_actual_query_distances.csv',distances)
    print('Wrote follow-up summaries, three figures and actual MQAR query distances.')


if __name__=='__main__':main()

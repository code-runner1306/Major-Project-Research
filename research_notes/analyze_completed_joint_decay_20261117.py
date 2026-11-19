"""Audit uploaded predictions and summarize completed joint/boundary runs."""
from collections import defaultdict,Counter
import csv
import hashlib
import json
from pathlib import Path
import statistics
import subprocess
import sys

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch
from evaluate_decay_boundary import cases_for,pair_metrics
from run_joint_recipe_study import identity as joint_identity

EVAL=Path('research_results/dynamic_decay_boundary_v1/20261117T154211097_be1684fd')
JOINT=Path('output/joint_recipe_study/joint-timescale-sensitivity-v1/20261117T095630656_b951c742')
OLD=Path('output/calibrated_recipe_study/matched-decay-v1')
OUT=Path('research_results/completed_joint_decay_review_20261117')


def save_csv(name,rows):
    with (OUT/name).open('w',newline='',encoding='utf-8') as stream:
        w=csv.DictWriter(stream,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)


def main():
    torch.set_num_threads(1);OUT.mkdir(parents=True,exist_ok=True)
    report=json.loads((EVAL/'results.json').read_text());protocol=json.loads((EVAL/'protocol.json').read_text())
    assert report['status']=='complete' and len(report['results'])==540 and not protocol['cpu_smoke']
    assert protocol['examples']==512 and protocol['fit_examples']==256 and protocol['seed_offset']==700000
    for name,digest in protocol['evaluator_sources'].items():
        data=subprocess.check_output(['git','show','e860526:'+name]).replace(b'\r\n',b'\n')
        assert hashlib.sha256(data).hexdigest()==digest,(name,'evaluator source mismatch')
    rows=report['results'];seen=set();predictions={};cases={};stale_rows=[]
    for family in ('mqar','overwrite'):
        for seed in (42,43,44):cases[family,seed]=cases_for(family,seed,512)
    for row in rows:
        key=tuple(row[k] for k in ('family','seed','initialization','variant','intervention','cell'))
        assert key not in seen;seen.add(key)
        family,seed,init,variant,mode,cell=key;case=cases[family,seed][cell]
        name=f'{family}_seed{seed}_{init}_{variant}_{mode}_{cell}.npz'
        path=EVAL/'predictions'/name
        assert hashlib.sha256(path.read_bytes()).hexdigest()==row['prediction_sha256']
        mask=case.data.labels!=-100
        with np.load(path) as data:
            pred=torch.from_numpy(data['predictions'].copy())
            assert np.array_equal(data['labels'],case.data.labels[mask].numpy())
            assert np.array_equal(data['kinds'],case.data.kinds[mask].numpy())
            assert np.array_equal(data['example_index'],mask.nonzero()[:,0].numpy())
        predictions[key]=pred
        assert row['metrics']['data_sha256']==case.data.identity['sha256']
        correct=pred==case.data.labels[mask];kinds=case.data.kinds[mask]
        assert int(correct.sum())==row['metrics']['correct'] and len(correct)==row['metrics']['total']
        assert abs(float(correct.double().mean())-row['metrics']['exact_query_accuracy'])<1e-10
        if family=='overwrite':
            latest=kinds==1;ret=kinds==2;prior=case.stale_values[mask]
            assert abs(float(correct[latest].double().mean())-row['metrics']['latest_value_accuracy'])<1e-10
            assert abs(float(correct[ret].double().mean())-row['metrics']['unrelated_association_retention'])<1e-10
            matches=(pred[:,None]==prior)&(prior>=0)
            assert abs(float(matches.any(-1)[latest].double().mean())-row['metrics']['stale_value_error_rate'])<1e-7
            if cell=='rewrites3' and mode=='original':
                counts=Counter()
                for prediction,target,olds in zip(pred[latest].tolist(),case.data.labels[mask][latest].tolist(),prior[latest].tolist()):
                    if prediction==target:counts['correct']+=1
                    elif prediction in olds:counts[['initial','first_replacement','second_replacement'][olds.index(prediction)]]+=1
                    else:counts['other']+=1
                stale_rows.append(dict(seed=seed,initialization=init,variant=variant,
                    **{k:counts[k] for k in ('correct','initial','first_replacement','second_replacement','other')}))
    for row in rows:
        if not row['cell'].startswith('pair_'):continue
        key=tuple(row[k] for k in ('family','seed','initialization','variant','intervention','cell'))
        f,s,i,v,m,cell=key;kind=cell[5:];base=cases[f,s]['rewrites3'];changed=cases[f,s][cell]
        actual=pair_metrics(base,changed,predictions[f,s,i,v,m,'rewrites3'],predictions[key],kind)
        assert actual==row['paired_metrics'],key
    # Exact selected evaluation matrix, not merely a matching row count.
    expected=set()
    for f in ('mqar','overwrite'):
        for s in (42,43,44):
            for i in ('default','calibrated','joint'):
                for v in ('scalar','channel-wise'):
                    for m in ('original','static','mean') if v=='channel-wise' else ('original',):
                        expected.update((f,s,i,v,m,cell) for cell in cases[f,s])
    assert seen==expected
    for path in (EVAL/'audits').glob('*.json'):
        audit=json.loads(path.read_text())
        if 'passed' in audit:assert audit['passed']
        if 'cases' in audit:
            for cell,data in audit['cases'].items():
                assert data['train_overlap']==0 and data['audit']['passed']
                assert data['identity']['sha256']==cases[audit['family'],audit['seed']][cell].data.identity['sha256']
    grouped=defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in ('family','initialization','variant','intervention','cell'))].append(row)
    averages=[]
    for key,group in sorted(grouped.items()):
        assert len(group)==3
        metrics={name:(statistics.mean(r['metrics'][name] for r in group) if group[0]['metrics'][name] is not None else None)
            for name in ('exact_query_accuracy','latest_value_accuracy','unrelated_association_retention','stale_value_error_rate')}
        averages.append(dict(zip(('family','initialization','variant','intervention','cell'),key),**metrics))
    save_csv('condition_means.csv',averages);save_csv('obsolete_value_breakdown.csv',stale_rows)
    training=[]
    for root in (OLD,JOINT):
        for path in root.rglob('summary.json'):
            summary=json.loads(path.read_text());c=summary.get('settings',{})
            if c.get('stage')!='explore' or c.get('seed') not in (42,43,44):continue
            assert summary['status']=='complete' and summary['accepted_optimizer_steps']==3136
            if root==JOINT:
                assert c['source_hashes']==joint_identity() and c['epochs']==16
                assert len(json.loads((path.parent/'development_curve.json').read_text()))==16
                cal=json.loads((path.parent/'calibration.json').read_text());assert cal['passed']
                assert max(max(x['sensitivity_error'],x['log_decay_error']) for x in cal['layers'])<1e-6
                assert json.loads((path.parent/'calibrated_backend_qualification.json').read_text())['passed']
            training.append(dict(family=c['family'],seed=c['seed'],initialization=c['initialization'],variant=c['variant'],
                accuracy_aulc=summary['learning']['accuracy_aulc_epochs1_16'],retention_aulc=summary['learning']['retention_aulc_epochs1_16'],
                final_accuracy=summary['final_development']['answer_free']['exact_query_accuracy'],wall_seconds=summary['wall_seconds']))
    assert len(training)==36;save_csv('training_comparison.csv',training)
    contrasts=json.loads((EVAL/'contrasts.json').read_text())
    assert len(contrasts)==18
    audit=dict(remote_commit='2b3d0af',evaluation_count=540,unique_checkpoints=36,raw_prediction_hashes_verified=540,
        data_cases_regenerated=len([c for group in cases.values() for c in group]),
        paired_metric_rows_recomputed=sum(r['cell'].startswith('pair_') for r in rows),
        primary_seed_rules_passed=sum(r['seed_operational_reliance'] for r in contrasts),joint_training_reports_verified=12,
        evaluation_wall_seconds=report['wall_seconds'],environment=json.loads((EVAL/'environment.json').read_text()),
        joint_session=json.loads((JOINT/'session_status.json').read_text()),
        limitation='No joint checkpoint weights uploaded; prediction, data, source, report audits do not independently rerun inference')
    (OUT/'verification.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in audit.items() if k not in ('joint_session','environment')},indent=2))
    for init in ('default','calibrated','joint'):
        a=[r for r in stale_rows if r['initialization']==init]
        print('STALE',init,{k:sum(r[k] for r in a) for k in ('correct','initial','first_replacement','second_replacement','other')})
    make_figures(averages)


def make_figures(rows):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(12,4.3))
    inits=['default','calibrated','joint'];colors=['#3066be','#de8f05']
    for ax,family,cell,metric in [(axes[0],'mqar','load12','exact_query_accuracy'),
                                  (axes[1],'overwrite','rewrites3','latest_value_accuracy')]:
        for j,mode in enumerate(['original','static']):
            values=[100*next(r[metric] for r in rows if (r['family'],r['initialization'],r['variant'],r['intervention'],r['cell'])==(family,i,'channel-wise',mode,cell)) for i in inits]
            ax.bar(np.arange(3)+(j-.5)*.34,values,width=.34,label='Dynamic' if mode=='original' else 'Fixed per channel',color=colors[j])
        ax.set_xticks(np.arange(3),['Default','Mean-alpha','Joint']);ax.set_ylim(0,105);ax.set_ylabel('Accuracy (%)')
        ax.set_title('MQAR: 12 associations, length 64' if family=='mqar' else 'Overwrite: latest value after 3 rewrites')
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    axes[0].legend(loc='lower left');fig.suptitle('Held-out dynamic vs fixed-channel decay: means over 3 seeds')
    fig.tight_layout();fig.savefig(OUT/'dynamic_vs_static.png',dpi=170);plt.close(fig)


if __name__=='__main__':main()

"""Rebuild the checked three-seed exploration tables and figures from saved runs."""
import csv
import json
from pathlib import Path
import statistics

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

ROOT=Path('output/calibrated_recipe_study/matched-decay-v1')
OUT=Path(__file__).resolve().parent
CONDITIONS=[('default','scalar'),('default','channel-wise'),('calibrated','scalar'),('calibrated','channel-wise')]
COLORS=['#284b63','#2a9d8f','#b36a32','#a23b72']
runs={}
for path in ROOT.rglob('summary.json'):
    summary=json.loads(path.read_text())
    cfg=summary['settings']
    if cfg['stage']!='explore' or cfg['seed'] not in [42,43,44]:continue
    key=cfg['family'],cfg['seed'],cfg['initialization'],cfg['variant']
    if key in runs:raise ValueError(f'Duplicate: {key}')
    assert summary['status']=='complete' and summary['accepted_optimizer_steps']==3136
    assert cfg['epochs']==16 and cfg['scheduler_epochs']==64
    with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
        checkpoint=torch.load(path.parent/'latest_checkpoint.pt',map_location='cpu',weights_only=True)
    assert checkpoint['completed_steps']==3136 and checkpoint['completed_epochs']==16
    assert checkpoint['settings']==cfg
    del checkpoint
    audit=json.loads((path.parent/'data_audit.json').read_text())
    assert audit['passed'] and audit['train_development_content_overlap']==0
    runs[key]=(path,summary,json.loads((path.parent/'development_curve.json').read_text()))
assert len(runs)==24
rows=[];contrasts=[];aggregates=[]
for family in ['mqar','overwrite']:
    metric='accuracy_aulc_epochs1_16' if family=='mqar' else 'retention_aulc_epochs1_16'
    for seed in [42,43,44]:
        for initialization,variant in CONDITIONS:
            path,s,curve=runs[family,seed,initialization,variant]
            final=s['final_development']['answer_free']
            rows.append(dict(family=family,seed=seed,initialization=initialization,variant=variant,
                steps=s['accepted_optimizer_steps'],primary_aulc=s['learning'][metric],
                final_accuracy=final['exact_query_accuracy'],final_retention=final['unrelated_association_retention'],
                latest_accuracy=final['latest_value_accuracy'],stale_error=final['stale_value_error_rate'],
                threshold_first_step=s['learning']['first_step_of_three_consecutive_95'],
                training_minutes=s['wall_seconds']/60,source=str(path)))
        def score(i,v):return runs[family,seed,i,v][1]['learning'][metric]
        default_gap=score('default','channel-wise')-score('default','scalar')
        calibrated_gap=score('calibrated','channel-wise')-score('calibrated','scalar')
        contrasts.append(dict(family=family,seed=seed,
            scalar_calibration_effect=score('calibrated','scalar')-score('default','scalar'),
            channel_calibration_effect=score('calibrated','channel-wise')-score('default','channel-wise'),
            default_channel_minus_scalar=default_gap,calibrated_channel_minus_scalar=calibrated_gap,
            interaction=calibrated_gap-default_gap))
    for initialization,variant in CONDITIONS:
        ss=[runs[family,seed,initialization,variant][1] for seed in [42,43,44]]
        values=[s['learning'][metric] for s in ss]
        aggregates.append(dict(family=family,initialization=initialization,variant=variant,
            primary_mean=statistics.mean(values),primary_sample_sd=statistics.stdev(values),
            final_accuracy_mean=statistics.mean(s['final_development']['answer_free']['exact_query_accuracy'] for s in ss)))
for name,values in [('models.csv',rows),('paired_contrasts.csv',contrasts),('condition_means.csv',aggregates)]:
    with (OUT/name).open('w',newline='',encoding='utf-8') as file:
        writer=csv.DictWriter(file,fieldnames=values[0].keys());writer.writeheader();writer.writerows(values)
(OUT/'analysis.json').write_text(json.dumps(dict(models=rows,contrasts=contrasts,means=aggregates,
    interpretation='Exploratory development metrics; three paired model/data seeds, no new confirmation'),indent=2),encoding='utf-8')

fig,axes=plt.subplots(2,3,figsize=(13,7),sharex=True,sharey=True)
for row,family in enumerate(['mqar','overwrite']):
    for col,seed in enumerate([42,43,44]):
        ax=axes[row,col]
        metric='exact_query_accuracy' if family=='mqar' else 'unrelated_association_retention'
        for (initialization,variant),color in zip(CONDITIONS,COLORS):
            curve=runs[family,seed,initialization,variant][2]
            ax.plot([x['epoch'] for x in curve],[100*x['answer_free'][metric] for x in curve],
                    color=color,label=f'{variant}, {initialization}',linewidth=1.8)
        ax.set_title(f'{family.upper()} · seed {seed}')
        ax.set_ylim(0,102);ax.set_xlim(1,16);ax.grid(alpha=.2)
        if row==1:ax.set_xlabel('Epoch (64-epoch schedule, stopped at 16)')
        if col==0:ax.set_ylabel('Query accuracy (%)' if family=='mqar' else 'Unrelated retention (%)')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='lower center',ncol=4,frameon=False)
fig.suptitle('All three seeds: final accuracy can hide learning-speed differences',fontsize=14)
fig.tight_layout(rect=(0,.07,1,.95));fig.savefig(OUT/'learning_curves.png',dpi=180);plt.close(fig)

fig,axes=plt.subplots(1,2,figsize=(10,4.2),sharey=True)
for ax,family in zip(axes,['mqar','overwrite']):
    cs=[r for r in contrasts if r['family']==family]
    ax.axhline(0,color='#777',linewidth=.8)
    for seed,color in zip([42,43,44],['#284b63','#2a9d8f','#b36a32']):
        c=next(x for x in cs if x['seed']==seed)
        ax.plot([0,1],[c['scalar_calibration_effect']*100,c['channel_calibration_effect']*100],
                'o-',color=color,label=f'Seed {seed}',linewidth=1.2)
    ax.set_xticks([0,1],['Scalar','Channel-wise']);ax.set_title(family.upper());ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel('Calibration effect on primary AULC (percentage points)')
axes[1].legend(frameon=False)
fig.suptitle('Above zero: calibration helped learning; below zero: it hurt')
fig.tight_layout();fig.savefig(OUT/'calibration_effects.png',dpi=180);plt.close(fig)
print('Verified 24 final checkpoints and audits. Wrote three tables, JSON, and two figures to',OUT)

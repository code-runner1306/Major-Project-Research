"""Summarize uploaded joint-run reports; no checkpoint evaluation or training."""
import csv
import json
from pathlib import Path
import statistics

ROOT=Path('output/joint_recipe_study/joint-timescale-sensitivity-v1/20261117T095630656_b951c742')
LEGACY=Path('output/calibrated_recipe_study/matched-decay-v1')
OUT=Path('research_results/joint_partial_review_20261117')


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    joint_protocol=json.loads((ROOT/'study_protocol.json').read_text())
    rows=[];selected={};curves={}
    for root,protocol in [(LEGACY,'matched-decay-v1'),(ROOT,'joint-timescale-sensitivity-v1')]:
        for path in root.rglob('settings.json'):
            cfg=json.loads(path.read_text())
            if cfg.get('protocol')!=protocol or cfg.get('stage')!='explore':continue
            summary_path=path.parent/'summary.json';curve_path=path.parent/'development_curve.json'
            if not curve_path.exists():continue
            curve=json.loads(curve_path.read_text());key=(cfg['family'],cfg['seed'],cfg['initialization'],cfg['variant'])
            if key in selected:raise ValueError('Duplicate report condition')
            selected[key]=path;curves[key]=curve
            if [row['epoch'] for row in curve]!=list(range(1,len(curve)+1)):raise ValueError('Nonmonotonic curve')
            if any(row['step']!=196*row['epoch'] for row in curve):raise ValueError('Unexpected optimizer steps')
            summary=json.loads(summary_path.read_text()) if summary_path.exists() else None
            if summary and (summary['status']!='complete' or summary['accepted_optimizer_steps']!=3136 or len(curve)!=16):raise ValueError('Incomplete final report')
            metrics=curve[-1]['answer_free'];learning=summary['learning'] if summary else {}
            rows.append(dict(family=key[0],seed=key[1],initialization=key[2],variant=key[3],
                complete=bool(summary),epochs=len(curve),steps=curve[-1]['step'],
                accuracy=metrics['exact_query_accuracy'],latest=metrics['latest_value_accuracy'],retention=metrics['unrelated_association_retention'],
                accuracy_aulc=learning.get('accuracy_aulc_epochs1_16'),retention_aulc=learning.get('retention_aulc_epochs1_16'),
                wall_seconds=summary['wall_seconds'] if summary else None,path=str(path.parent)))
    for key,path in selected.items():
        if key[2]!='joint':continue
        old=selected[(key[0],key[1],'default',key[3])]
        cfg=json.loads(path.read_text());old_cfg=json.loads(old.read_text())
        fields=('family','seed','variant','epochs','scheduler_epochs','vocabulary','length','width','train_examples',
            'validation_examples','effective_batch','microbatch','learning_rate','precision')
        if any(cfg[field]!=old_cfg[field] for field in fields):raise ValueError('Joint recipe differs beyond initialization/source identity')
        for name in ['data_manifests.json','initialization_manifest.json']:
            left=json.loads((path.parent/name).read_text());right=json.loads((old.parent/name).read_text())
            if left!=right:raise ValueError(f'Base initialization/data mismatch: {path} {name}')
        calibration=json.loads((path.parent/'calibration.json').read_text())
        if not calibration['passed'] or max(max(x['sensitivity_error'],x['log_decay_error']) for x in calibration['layers'])>1e-6:raise ValueError('Joint fit failed')
        if not json.loads((path.parent/'calibrated_backend_qualification.json').read_text())['passed']:raise ValueError('Backend check failed')
    with (OUT/'comparison.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    pairs=[json.loads(p.read_text()) for p in ROOT.rglob('pair_status.json')]
    durations={family:[p['wall_seconds'] for p in pairs if p['family']==family and p['status']=='complete'] for family in ['mqar','overwrite']}
    mqar_remaining=max((16-len(curves['mqar',44,'joint',v]))/16*statistics.mean(durations['mqar']) for v in ['scalar','channel-wise'])
    overwrite_remaining=statistics.mean(durations['overwrite'])
    eta=dict(snapshot_commit='226235b',snapshot_commit_time_ist='2026-11-17 14:24:10 +05:30',
        completed_joint_models=sum(r['complete'] for r in rows if r['initialization']=='joint'),
        partial_mqar_seed44_epochs={v:len(curves['mqar',44,'joint',v]) for v in ['scalar','channel-wise']},
        pending_models=['overwrite/seed44/scalar','overwrite/seed44/channel-wise'],
        measured_pair_seconds=durations,remaining_mqar_seconds=mqar_remaining,
        remaining_overwrite_seconds=overwrite_remaining,total_remaining_seconds_from_uploaded_epochs=mqar_remaining+overwrite_remaining,
        projected_total_six_pairs_seconds=3*sum(statistics.mean(t) for t in durations.values()),
        caveats=['Snapshot, not live monitoring','Partial console logs trail saved development curves; not an atomic export',
            'Assumes uninterrupted run, same throughput, and no concurrent GPU evaluation',
            'No uploaded model checkpoint files; this review uses reports only'])
    (OUT/'runtime_estimate.json').write_text(json.dumps(eta,indent=2),encoding='utf-8')
    (OUT/'learning_curves.json').write_text(json.dumps({'/'.join(map(str,key)):value for key,value in curves.items()},indent=2),encoding='utf-8')
    print(json.dumps(eta,indent=2))


if __name__=='__main__':main()

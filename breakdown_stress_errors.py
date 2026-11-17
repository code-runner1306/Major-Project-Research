"""Post-hoc error types for already evaluated stress cells; no new conditions."""
import json
from pathlib import Path
import torch
from confirm_recipe_study import select_runs
from recipe_model import RecipeLM
from stress_recipe_checkpoints import make_mqar,make_overwrite,evaluate_case


def predict(model,case):
    original=model.scored_logits;outputs=[]
    def capture(tokens,mask):
        logits=original(tokens,mask);outputs.append(logits.argmax(-1).detach().cpu());return logits
    model.scored_logits=capture
    try:metrics=evaluate_case(model,case)
    finally:delattr(model,'scored_logits')
    return metrics,torch.cat(outputs)


def main():
    root=Path('research_results/memory_stress_tests_20261117')
    stress=json.loads((root/'results.json').read_text());assert stress['status']=='complete'
    selected=select_runs(Path('output/calibrated_recipe_study/matched-decay-v1'),['mqar','overwrite'],[42,43,44],[],'explore')
    torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False
    report=dict(scope='post-hoc breakdown of existing more_keys seed44 MQAR and more_rewrites overwrite cases',mqar=[],overwrite=[])
    chosen=[(f,seed,i,v) for f,seed,i,v in selected if f=='overwrite' or seed==44]
    for family,seed,init,variant in chosen:
        path,summary=selected[family,seed,init,variant]
        model=RecipeLM(variant,8192,64,128,'chunked',seed)
        with torch.serialization.safe_globals([torch.torch_version.TorchVersion]):
            saved=torch.load(path.parent/'latest_checkpoint.pt',map_location='cpu',weights_only=True)
        model.load_state_dict(saved['model'],strict=True);del saved
        case=make_mqar(256,seed+500000,64,8) if family=='mqar' else make_overwrite(256,seed+500000,4,8,3,seed+100000)
        metrics,predictions=predict(model,case)
        expected=next(x for x in stress['results'] if (x['family'],x['seed'],x['initialization'],x['variant'],x['cell'])==
                      (family,seed,init,variant,'more_keys' if family=='mqar' else 'more_rewrites'))
        assert metrics==expected['metrics']
        mask=case.data.labels!=-100
        if family=='mqar':
            correct=predictions==case.data.labels[mask]
            ordinal=[]
            for tokens,labels in zip(case.data.tokens.tolist(),case.data.labels.tolist()):
                positions={tokens[2*i]:i for i in range(8)}
                ordinal.extend(positions[tokens[pos]] for pos,target in enumerate(labels) if target!=-100)
            ordinal=torch.tensor(ordinal)
            report['mqar'].append(dict(seed=seed,initialization=init,variant=variant,
                by_memory_ordinal=[dict(memory_ordinal=i+1,correct=int(correct[ordinal==i].sum()),total=int((ordinal==i).sum())) for i in range(8)]))
        else:
            latest=case.data.kinds[mask]==1;targets=case.data.labels[mask][latest]
            pred=predictions[latest];prior=case.stale_values[mask][latest]
            old_counts=[int((pred==prior[:,i]).sum()) for i in range(3)]
            correct=int((pred==targets).sum())
            report['overwrite'].append(dict(seed=seed,initialization=init,variant=variant,total_latest=256,
                correct_latest=correct,initial_value_errors=old_counts[0],first_rewrite_errors=old_counts[1],
                second_rewrite_errors=old_counts[2],other_errors=256-correct-sum(old_counts)))
        print(f'Error breakdown: {family}/seed{seed}/{init}/{variant}',flush=True)
        (root/'error_breakdown.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':main()

"""Batched retrieval evaluation with the existing answer-erasure contract."""
import torch
from torch.nn import functional as F


@torch.no_grad()
def evaluate(model,data,device,batch=32,history='answer_free'):
    if history not in ('answer_free','teacher_forced') or batch<1:
        raise ValueError('Invalid evaluation settings')
    was_training=model.training
    model.eval()
    totals={name:[0,0] for name in ('all','first','later','latest','unrelated')}
    loss_sum=0.
    complete=stale_errors=0
    try:
        for start in range(0,len(data.tokens),batch):
            stop=start+batch
            tokens=data.tokens[start:stop].clone()
            if history=='answer_free':
                # Match retrieval.evaluate_retrieval exactly: erase all gold
                # answer inputs before the pass, rather than feeding predictions.
                tokens[data.erase_answers[start:stop]]=1
            labels=data.labels[start:stop]
            mask=labels!=-100
            selected=model.scored_logits(tokens.to(device),mask.to(device))
            targets=labels[mask].to(device)
            loss_sum+=F.cross_entropy(selected,targets,reduction='sum').item()
            predictions=selected.argmax(-1).cpu()
            target_flat=labels[mask]
            correct=predictions==target_flat
            counts=mask.sum(1)
            offset=0
            for count in counts.tolist():
                complete+=int(correct[offset:offset+count].all())
                offset+=count
            ordinals=mask.cumsum(1)-1
            kinds=data.kinds[start:stop]
            group_masks={'all':torch.ones_like(correct),'first':ordinals[mask]==0,
                         'later':ordinals[mask]>0,'latest':kinds[mask]==1,'unrelated':kinds[mask]==2}
            for name,group in group_masks.items():
                totals[name][0]+=int(correct[group].sum())
                totals[name][1]+=int(group.sum())
            stale_errors+=int(((predictions==data.stale[start:stop][mask])&group_masks['latest']).sum())
    finally:
        model.train(was_training)
    def accuracy(name):
        hits,total=totals[name]
        return hits/total if total else None
    correct,total=totals['all']
    if not total:
        raise ValueError('No supervised answers to evaluate')
    return {'exact_query_accuracy':accuracy('all'),'correct':correct,'total':total,
        'loss':loss_sum/total,'first_query_accuracy':accuracy('first'),
        'later_query_accuracy':accuracy('later'),'latest_value_accuracy':accuracy('latest'),
        'unrelated_association_retention':accuracy('unrelated'),
        'stale_value_error_rate':stale_errors/totals['latest'][1] if totals['latest'][1] else None,
        'example_accuracy':complete/len(data.tokens),'evaluated_examples':len(data.tokens),
        'history_conditioning':history,'data_sha256':data.identity['sha256']}

"""Memory-bounded, byte-equivalent generator for pinned Zoology MQAR."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
import numpy as np
import torch

ZOOLOGY_REVISION = 'de4e258784224e09909c257ff3ea040f089ed660'


def mqar(examples=100000, vocabulary=8192, length=64, pairs=4, seed=42):
    """Same RNG call order as upstream _mqar(random_non_queries=False).

    Upstream tiles the entire key/value vocabulary for every row. Each row
    passed to np.random.choice is identical; reusing that immutable row avoids
    >6GB of temporary arrays at100k examples without changing any RNG calls.
    A private RandomState reproduces upstream np.random.seed/choice and avoids
    changing the caller's global random state.
    """
    if examples < 1 or length % 2 or vocabulary <= length or pairs < 1 or 4*pairs > length:
        raise ValueError('Invalid pinned MQAR dimensions')
    rng = np.random.RandomState(seed)
    key_choices = np.arange(1,vocabulary//2)
    value_choices = np.arange(vocabulary//2,vocabulary)
    keys = np.stack([rng.choice(key_choices,replace=False,size=pairs) for _ in range(examples)])
    values = np.stack([rng.choice(value_choices,replace=False,size=pairs) for _ in range(examples)])
    context = 2*pairs
    memory = np.zeros((examples,context),dtype=np.int64)
    memory[:,0::2],memory[:,1::2]=keys,values
    space=(length-context)//2
    power=.01*np.arange(1,space+1)**(.01-1)
    power/=power.sum()
    candidates=np.arange(space,dtype=int)
    gaps=np.stack([rng.choice(candidates,replace=False,p=power,size=pairs) for _ in range(examples)])
    queries=np.zeros((examples,length-context+1),dtype=np.int64)
    np.put_along_axis(queries,gaps*2,keys,axis=1)
    sequence=np.concatenate([memory,queries],axis=1)
    labels=np.full((examples,length+1),-100,dtype=np.int64)
    np.put_along_axis(labels,gaps*2+context+1,values,axis=1)
    return torch.tensor(sequence[:,:-1]),torch.tensor(labels[:,1:])


def manifest(tokens,labels,**settings):
    digest=hashlib.sha256()
    for tensor in (tokens,labels):
        digest.update(tensor.contiguous().numpy().tobytes())
    return {'generator':'pinned-zoology-mqar-memory-bounded','upstream_revision':ZOOLOGY_REVISION,
            'sha256':digest.hexdigest(),'shape':list(tokens.shape),**settings}


def audit_mqar(tokens,labels,pairs=4):
    """Independent prefix dictionary and changed-query checks, no model labels as input."""
    if pairs<2:
        raise ValueError('Changed-query audit requires at least two keys')
    mismatches=counterfactual_failures=checks=0
    first_correct=last_correct=0
    for row,target in zip(tokens.tolist(),labels.tolist()):
        memory={row[index]:row[index+1] for index in range(0,2*pairs,2)}
        if len(memory)!=pairs:
            raise ValueError('MQAR context has duplicate keys')
        for position,answer in enumerate(target):
            if answer == -100:
                continue
            checks+=1
            mismatches+=int(memory.get(row[position])!=answer or position<2*pairs)
            other=next(key for key in memory if key!=row[position])
            counterfactual_failures+=int(memory[other]==answer)
            first_correct+=int(row[1]==answer)
            last_correct+=int(row[2*pairs-1]==answer)
    return {'passed':checks==len(tokens)*pairs and mismatches==0 and counterfactual_failures==0,
        'checks':checks,'label_mismatches':mismatches,'counterfactual_failures':counterfactual_failures,
        'key_free_first_value_accuracy':first_correct/max(1,checks),
        'key_free_last_value_accuracy':last_correct/max(1,checks),
        'scope':'specified label/query-key/copy checks only; not trained learnability'}


@dataclass
class RecipeData:
    tokens: torch.Tensor
    labels: torch.Tensor
    kinds: torch.Tensor  # -1 ignored,0 MQAR,1 latest,2 unrelated.
    stale: torch.Tensor
    erase_answers: torch.Tensor
    identity: dict


def build_data(family,examples,seed,split='train',vocabulary=8192,length=64):
    if family=='mqar':
        tokens,labels=mqar(examples,vocabulary,length,4,seed)
        return RecipeData(tokens,labels,torch.where(labels!=-100,0,-1),
            torch.full_like(labels,-1),torch.zeros_like(tokens,dtype=torch.bool),
            manifest(tokens,labels,seed=seed,split=split))
    if family!='overwrite':
        raise ValueError('family must be mqar or overwrite')
    from config import RetrievalConfig
    from retrieval import RetrievalDataset
    config=RetrievalConfig(vocabulary_size=vocabulary,sequence_lengths=(length+1,),
        key_counts=(4,),query_distances=(8,),overwrite_counts=(1,),generator_version='mqar-overwrite-v4')
    dataset=RetrievalDataset(task='overwrite',examples=examples,config=config,seed=seed,split=split)
    records=[dataset.example(i) for i in range(examples)]
    tokens=torch.stack([x.tokens for x in records])
    labels=torch.stack([x.labels for x in records])
    kinds=torch.full_like(labels,-1)
    stale=torch.full_like(labels,-1)
    erase=torch.zeros_like(tokens,dtype=torch.bool)
    for index,record in enumerate(records):
        for answer in record.answers:
            kinds[index,answer.label_position]=1 if answer.kind=='latest' else 2
            if answer.kind=='latest':
                if len(answer.stale_values)!=1:
                    raise ValueError('Recipe extension requires exactly one rewrite')
                stale[index,answer.label_position]=answer.stale_values[0]
            if answer.label_position+1<tokens.size(1):
                erase[index,answer.label_position+1]=True
    identity=manifest(tokens,labels,seed=seed,split=split)
    identity.update(generator='audited-v4-overwrite-extension',upstream_revision=None,
                    source_manifest=dataset.manifest(),emitted_sequence_length=length+1,
                    model_input_length=length)
    return RecipeData(tokens,labels,kinds,stale,erase,identity)


def split_overlap(train,development):
    """Check actual emitted cases, not only unequal seed/manifest labels."""
    def hashes(data):
        rows=np.concatenate((data.tokens.numpy(),data.labels.numpy()),axis=1)
        return {hashlib.sha256(row.tobytes()).digest() for row in rows}
    return len(hashes(train).intersection(hashes(development)))

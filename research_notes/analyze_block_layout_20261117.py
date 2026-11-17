"""Accuracy split by memory layout: is the target block first (post_target_write_count == 3)?
Runs the 12 local legacy overwrite checkpoints on the write-gate stress cells (offset 500000). CPU."""
import sys, torch
sys.path.insert(0,'.')
from pathlib import Path
from collections import Counter
from confirm_recipe_study import select_runs
from analyze_write_gate import build_cell, classify, load
from retrieval import RetrievalDataset
from config import RetrievalConfig
torch.set_num_threads(8)
sel=select_runs(Path('output/calibrated_recipe_study/matched-decay-v1'),['overwrite'],[42,43,44],[],'explore')
def post(seed,R):
    cfg=RetrievalConfig(vocabulary_size=8192,sequence_lengths=(65,),key_counts=(4,),query_distances=(8,),overwrite_counts=(R,),generator_version='mqar-overwrite-v4')
    d=RetrievalDataset(task='overwrite',examples=256,config=cfg,seed=seed+500000,split='held_out')
    return [d.example(i).metadata['post_target_write_count']==3 for i in range(256)]
# training distribution: how often is the target block first (post==3) with one rewrite?
cfg=RetrievalConfig(vocabulary_size=8192,sequence_lengths=(65,),key_counts=(4,),query_distances=(8,),overwrite_counts=(1,),generator_version='mqar-overwrite-v4')
d=RetrievalDataset(task='overwrite',examples=2000,config=cfg,seed=42+100000,split='train')
print('train post_target_write_count dist:',Counter(d.example(i).metadata['post_target_write_count'] for i in range(2000)))
for seed in (42,43,44):
    for R in (2,3,4):
        cell=build_cell(256,seed,R); mask=cell['case'].data.labels!=-100; block=post(seed,R)
        for key,(path,_) in sorted(sel.items()):
            if key[1]!=seed: continue
            m=load(path,key[3],seed)
            with torch.no_grad(): pred=m.scored_logits(cell['tokens'],mask).argmax(-1)
            o,_=classify(pred,cell)
            inb=[x for x,b in zip(o,block) if b]; outb=[x for x,b in zip(o,block) if not b]
            print(seed,R,key[2][:4],key[3][:4],f"block {sum(x=='correct' for x in inb)}/{len(inb)} errs={dict(Counter(x for x in inb if x!='correct'))}",
                  f"| nonblock {sum(x=='correct' for x in outb)}/{len(outb)} errs={dict(Counter(x for x in outb if x!='correct'))}")

"""Summarize final gate probes without claiming an unverified initial replay."""
import json
from pathlib import Path
import statistics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
s=json.loads((ROOT/'gate_diagnostics.json').read_text())
assert s['status']=='complete' and s['models_completed']==24
rows=s['measurements'];means=[]
for family in ['mqar','overwrite']:
    for init in ['default','calibrated']:
        for variant in ['scalar','channel-wise']:
            for layer in range(4):
                selected=[x for x in rows if (x['family'],x['initialization'],x['variant'],x['layer'],x['corpus'])==
                          (family,init,variant,layer,'calibration_training')]
                assert len(selected)==3
                means.append(dict(family=family,initialization=init,variant=variant,layer=layer+1,
                    alpha_mean=statistics.mean(x['alpha']['mean'] for x in selected),
                    alpha_sample_sd=statistics.stdev(x['alpha']['mean'] for x in selected),
                    beta_mean=statistics.mean(x['beta']['mean'] for x in selected),
                    absolute_offset_sensitivity=statistics.mean(x['absolute_offset_sensitivity']['mean'] for x in selected),
                    channel_dispersion=statistics.mean(x['within_token_channel_std'] for x in selected)))
(ROOT/'condition_layer_means.json').write_text(json.dumps(means,indent=2),encoding='utf-8')

fig,axes=plt.subplots(1,2,figsize=(11,4.2),sharey=True)
for ax,family in zip(axes,['mqar','overwrite']):
    target=[2**(-1/h) for h in [16,32,64,128]]
    ax.plot([1,2,3,4],target,'k--',label='Recorded initial target',linewidth=1.5)
    for variant,color in [('scalar','#284b63'),('channel-wise','#a23b72')]:
        selected=[x for x in means if (x['family'],x['initialization'],x['variant'])==(family,'calibrated',variant)]
        ax.plot([x['layer'] for x in selected],[x['alpha_mean'] for x in selected],'o-',color=color,label='Trained '+variant)
        for seed in [42,43,44]:
            individual=sorted([x for x in rows if (x['family'],x['seed'],x['initialization'],x['variant'],x['corpus'])==
                               (family,seed,'calibrated',variant,'calibration_training')],key=lambda x:x['layer'])
            ax.scatter([x['layer']+1 for x in individual],[x['alpha']['mean'] for x in individual],color=color,alpha=.35,s=18)
    ax.set_title(family.upper());ax.set_xticks([1,2,3,4]);ax.set_xlabel('Layer');ax.grid(alpha=.2)
axes[0].set_ylabel('Mean decay multiplier (closer to 1 = less direct decay)')
axes[0].set_ylim(.8,1.005);axes[1].legend(fontsize=8,frameon=False)
fig.suptitle('Matched starting means do not remain matched after training')
fig.tight_layout();fig.savefig(ROOT/'calibrated_gate_means.png',dpi=180);plt.close(fig)
print('Wrote condition/layer means and calibrated gate plot.')

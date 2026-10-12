#!/usr/bin/env python3
from pathlib import Path
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('results/context_ensemble/metrics.csv'); d=pd.read_csv(p)
models=['Decision Tree','Random Forest','XGBoost','Meta Bagging','Gradient Boost','Hard Voting Ensemble']
w=d.set_index(['protocol','model'])
vals0=[100*w.loc[('unseen_recordings',name),'accuracy'] for name in models]
vals1=[100*w.loc[('within_record_windows',name),'accuracy'] for name in models]
fig,ax=plt.subplots(figsize=(11.4,5.1))
x=np.arange(len(models));bar=.36
ax.bar(x-bar/2,vals0,bar,label='Unseen recordings')
ax.bar(x+bar/2,vals1,bar,label='Windows from observed recordings')
ax.set_xticks(x,models,rotation=13,ha='right');ax.set_ylim(60,102)
ax.set_ylabel('Out-of-fold accuracy (%)');ax.set_title('EHG classification: same architecture, different evaluation populations')
ax.legend(loc='upper left');ax.spines[['top','right']].set_visible(False)
for i,(a,b) in enumerate(zip(vals0,vals1)):
    ax.annotate(f'{a:.1f}',(i-bar/2,a),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
    ax.annotate(f'{b:.1f}',(i+bar/2,b),xytext=(0,3),textcoords='offset points',ha='center',fontsize=8)
fig.tight_layout();out=Path('results/figures/context_ensemble.png');out.parent.mkdir(parents=True,exist_ok=True)
fig.savefig(out,dpi=170);plt.close(fig);print(out)

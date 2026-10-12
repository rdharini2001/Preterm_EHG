"""Produce a compact set of publication-friendly protocol comparison figures."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
res=ROOT/'results'
dest=res/'figures';dest.mkdir(parents=True,exist_ok=True)
win=pd.read_csv(res/'windows/metrics.csv')
keep=['1-NN Window Matching','Balanced Random Forest','Hist Gradient Boost','Random Forest']
f=win[win.model.isin(keep)].pivot(index='model',columns='evaluation',values='balanced_accuracy').reindex(keep)
fig,ax=plt.subplots(figsize=(9,4.2))
f.mul(100).plot(kind='bar',ax=ax,rot=13,width=.72)
ax.set_title('Independent-record vs overlapping-window validation',fontweight='bold',pad=13)
ax.set_ylabel('Balanced accuracy (%)');ax.set_xlabel('');ax.set_ylim(0,100)
ax.legend(['Record-disjoint CV','Window-stratified CV'],loc='upper right')
ax.grid(axis='y',alpha=.22);fig.tight_layout();fig.savefig(dest/'window_protocol_comparison.png',dpi=190);plt.close(fig)
adv=pd.read_csv(res/'advanced/metrics.csv').sort_values('balanced_accuracy')
fig,ax=plt.subplots(figsize=(9,4))
y=np.arange(len(adv));ax.barh(y,adv.balanced_accuracy*100)
ax.set_yticks(y,adv.model);ax.set_xlabel('Balanced accuracy (%)');ax.set_xlim(0,100)
ax.set_title('Nested train-only tuning | recording-level CV',fontweight='bold',pad=13)
ax.grid(axis='x',alpha=.22);fig.tight_layout();fig.savefig(dest/'advanced_balanced_accuracy.png',dpi=190);plt.close(fig)
print('Wrote results/figures/window_protocol_comparison.png and advanced_balanced_accuracy.png')

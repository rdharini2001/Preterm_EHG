"""Generate neural network comparison figure from actual holdout predictions."""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

p=Path(__file__).resolve().parents[1]
m=pd.read_csv(p/'results/tables/deep_metrics.csv')
m=m[m.model.isin(['CNN Ensemble','CNN 03','LSTM 20','LSTM 50','LSTM 100','LSTM 200','Sparse AE (16 units)'])]
fig,ax=plt.subplots(figsize=(10,4.8))
pos=list(range(len(m)));width=.34
ax.bar([i-width/2 for i in pos],m.accuracy*100,width,color='#356c90',label='Accuracy')
ax.bar([i+width/2 for i in pos],m.balanced_accuracy*100,width,color='#c5834b',label='Balanced accuracy')
ax.set(ylabel='Holdout classification (%)',ylim=(0,100),title='Neural-network comparison — unseen recordings',xticks=pos,xticklabels=m.model)
ax.legend(frameon=False)
fig.tight_layout();fig.savefig(p/'results/figures/deep_model_comparison.png',dpi=180);plt.close(fig)

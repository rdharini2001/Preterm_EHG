"""Saved-data visualization of the five-tree evaluation design search."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import numpy as np
import pandas as pd
r=Path(__file__).resolve().parents[1]; out=r/'results/figures';out.mkdir(parents=True,exist_ok=True)
s=pd.read_csv(r/'results/design_search/metrics.csv')
p=s[(s.model=='Hard Voting Ensemble')&(s.feature_view=='three_frequency_bands')].copy()
order=['record_cv','train_fold_oversampling','train_fold_smote','pre_cv_smote_diagnostic','pre_cv_oversampling_diagnostic']
labels=['Record CV','Train-only ROS','Train-only SMOTE','Pre-CV SMOTE*','Pre-CV ROS*']
p=p.set_index('evaluation').loc[order]
for key,title,file in [('accuracy','Prediction accuracy','design_accuracy.png'),
                        ('sensitivity','Preterm sensitivity','design_sensitivity.png')]:
    fig,ax=plt.subplots(figsize=(8.2,4.1))
    vals=p[key].to_numpy(float)*100
    bars=ax.bar(np.arange(len(vals)),vals,color=['#1b4965']*3+['#ae6d29']*2,width=.66)
    for b,v in zip(bars,vals):ax.annotate(f'{v:.1f}%',(b.get_x()+b.get_width()/2,v),xytext=(0,4),textcoords='offset points',ha='center',fontsize=9)
    ax.set_xticks(np.arange(len(vals)),labels,rotation=18,ha='right');ax.set_ylim(0,108)
    ax.set_ylabel('Percent');ax.set_title(f'{title} — five-tree hard voting')
    ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    fig.text(.02,.01,'* Diagnostic: synthetic examples were generated before validation splitting; not independent-pregnancy performance.',fontsize=7)
    fig.tight_layout(rect=[0,.055,1,1]);fig.savefig(out/file,dpi=170);plt.close(fig)

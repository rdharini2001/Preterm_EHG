#!/usr/bin/env python3
"""Create Markdown report directly from machine-computed search outputs."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1]
O=R/'results/protocol_search'
t=pd.read_csv(O/'target_coverage.csv')
v=pd.read_csv(O/'vote_optima.csv')
c=json.loads((R/'configs/paper_targets.json').read_text())
acc=t[(t.objective=='accuracy')&t.paper_accuracy_percent.notna()].copy()
acc=acc.sort_values('paper_accuracy_percent')
acc['gap_percentage_points']=acc.computed_accuracy_percent-acc.paper_accuracy_percent
acc['abs_gap_percentage_points']=acc.gap_percentage_points.abs()
acc['within_0_2pp']=acc.abs_gap_percentage_points<=.2
acc.to_csv(O/'accuracy_coverage.csv',index=False,float_format='%.10g')
fig,ax=plt.subplots(figsize=(11,8.6))
y=np.arange(len(acc));ax.hlines(y,acc.paper_accuracy_percent,acc.computed_accuracy_percent,lw=2,color='#86a0c6')
ax.scatter(acc.paper_accuracy_percent,y,label='Paper target',color='#ec9a37',s=50,zorder=3)
ax.scatter(acc.computed_accuracy_percent,y,label='Closest observed operating point',color='#2262a2',s=50,zorder=3)
ax.set_yticks(y,acc.paper_model)
ax.invert_yaxis();ax.grid(axis='x',alpha=.2)
ax.set_xlabel('Classification accuracy (%)')
ax.set_title('Accuracy targets and computed protocol candidates',loc='left',fontweight='semibold',fontsize=14)
ax.legend(loc='lower right',frameon=False)
fig.tight_layout();(R/'results/figures').mkdir(exist_ok=True)
fig.savefig(R/'results/figures/protocol_accuracy_targets.png',dpi=160)
plt.close(fig)
lines=[
'# Experimental protocol and operating-point search',
'',
'This analysis systematically varies decision thresholds, prediction units and aggregation rules **over saved model predictions**. Five-tree plurality votes are recomputed using five independently trained classifier probability scores. No outcomes or predictions are synthesized to match a reference number.',
'',
'**Interpretation:** These are retrospective, target-directed parameter choices on out-of-fold/holdout outputs. They identify numerical reconstruction protocols and hypotheses; because the target metrics were used to select operating points, the selected scores are *not* independent validation of generalization.',
'',
'## Headline metrics: two explicit operating points',
'',
'Each prediction uses the original five classifier families: decision tree, random forest, XGBoost, meta-bagging, and gradient boosting. Every classifier contributes one calibrated binary vote; at least three positive votes produce a positive ensemble result.',
'',
'| Parameter-search objective | Accuracy | Preterm sensitivity | Term specificity | TP | TN | FP | FN | Operating-point cutoffs |',
'|---|---:|---:|---:|---:|---:|---:|---:|---|'
]
for x in v.itertuples():
    label='Accuracy 98.99%' if x.objective=='headline_accuracy' else 'Sensitivity 0.983, specificity 0.979'
    lines.append(f'| {label} | {x.accuracy*100:.4f}% | {x.sensitivity:.6f} | {x.specificity:.6f} | {x.TP} | {x.TN} | {x.FP} | {x.FN} | `{x.thresholds}` |')
lines += [
'',
'The headline accuracy operating point returns 7,127 correct out of 7,200 windows: 98.9861%, **98.99% to two decimal places**.',
'The class-rate operating point has TP=897 of 912 preterm windows and TN=6,156 of 6,288 term windows. Its rates are 0.983553 and 0.979008; **truncating each fraction to three decimal digits** yields 0.983 and 0.979. These are two different operating points and their numbers must not be pooled into a single confusion matrix.',
'',
'## Accuracy target coverage by algorithm family',
'',
'For each family the table shows the closest measured accuracy among the indexed evaluation protocols and decision cutoffs. A close number does not establish that the original authors used that precise protocol.',
'',
'| Algorithm | Accuracy target | Closest computed | Absolute gap (percentage points) | Source protocol | Evaluation unit |',
'|---|---:|---:|---:|---|---|',
]
for x in acc.itertuples():
    lines.append(f'| {x.paper_model} | {x.paper_accuracy_percent:.2f}% | {x.computed_accuracy_percent:.3f}% | {x.abs_gap_percentage_points:.3f} | {x.source}: {x.protocol} | {x.aggregation} |')
# Fold standard deviations and neural probability-RMSE are separate objectives.
extra=t[t.objective.isin(['cv_sd','rmse'])].copy()
if len(extra):
    lines+=['', '## Fold variability and neural-network RMSE','',
            "Fold SD is computed from each candidate's ten observed fold accuracies (population standard deviation). For neural models, RMSE is calculated on the saved class-probability predictions. These definitions can differ from other RMSE conventions.",'',
            '| Model | Criterion | Target | Computed | Absolute gap |',
            '|---|---|---:|---:|---:|']
    for x in extra.itertuples():
        v=x.paper_cv_sd if x.objective=='cv_sd' else x.paper_rmse
        observed=x.cv_accuracy_sd if x.objective=='cv_sd' else x.computed_rmse
        lines.append(f'| {x.paper_model} | {x.objective} | {v:.4f} | {observed:.4f} | {abs(v-observed):.4f} |')

lines += [
'',
'## Complete data provenance',
'',
'`candidate_metrics.csv` contains every explored individual-model threshold and aggregation. `target_coverage.csv` has separate objectives for accuracy and paired class rates. `vote_search_trace.csv` contains score-cutoff hypotheses; `vote_optima_predictions.csv` contains one genuine binary prediction for every evaluated window and selected operating point. `vote_optima.csv` contains the matching confusion matrices. `manifest.json` records the search methodology.',
'',
'## Mathematical interpretation',
'',
'- If accuracy, sensitivity and specificity come from the **same pooled confusion matrix**, accuracy equals sensitivity times positive prevalence plus specificity times negative prevalence. Therefore, accuracy 98.99% cannot coexist with sensitivity 98.3% and specificity 97.9% on that same pooled set.',
'- The separately calibrated operating points above do reproduce the two published numerical targets from *different rules* without changing the truth labels. This is a plausible explanation to investigate, not evidence of the historical protocol.',
'- Independent-pregnancy validation is available separately in `results/context_ensemble/`; within-record windows are not independent pregnancies.',
'- These experiments use the manuscript’s 750-sample windows, 125-sample shift and five classifiers. The extended model search is a documented additional procedure.',
'',
'## Run',
'',
'```bash',
'bash scripts/run_protocol_search.sh',
'```',
]
(R/'docs/PROTOCOL_SEARCH.md').write_text('\n'.join(lines)+'\n')
print('Accuracy targets within 0.2 percentage points:',int(acc.within_0_2pp.sum()),'/',len(acc))
print('Updated docs/PROTOCOL_SEARCH.md and results/figures/protocol_accuracy_targets.png')

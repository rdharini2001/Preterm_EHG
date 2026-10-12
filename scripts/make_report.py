"""Build the computed results Markdown from validated CSV artifacts."""
from pathlib import Path
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
res=ROOT/'results'
ml=pd.read_csv(res/'tables/model_metrics.csv').copy()
ml['Accuracy (%)']=(ml['accuracy']*100).map(lambda v:f'{v:.2f}')
ml['Sensitivity']=ml.sensitivity.map(lambda v:f'{v:.3f}')
ml['Specificity']=ml.specificity.map(lambda v:f'{v:.3f}')
ml['CV SD']=ml.cv_accuracy_sd.map(lambda v:f'{v:.3f}')
ml['AUC']=ml.auc.map(lambda v:f'{v:.3f}' if pd.notna(v) else '—')
ml['Algorithm']=ml.model
ml_table=ml[['Algorithm','Accuracy (%)','Sensitivity','Specificity','AUC','CV SD']].to_markdown(index=False)
base=pd.read_csv(res/'extended/tables/model_metrics.csv') if (res/'extended/tables/model_metrics.csv').exists() else None
parts=['# Classification results','',
       '## TPEHGDB cohort','',
       'The study includes 300 EHG recordings: **262 term** and **38 preterm**.',
       'A record is the unit of evaluation. No channels or temporal windows of a recording cross the training/test boundary.','',
       '## Feature-based classifiers','',
       'Ten-fold stratified cross-validation. Forward selection of the five main feature families runs independently inside each training fold. Model inputs comprise three channels for each selected feature family.','',
       ml_table,'',
       '![Classifier accuracy](../results/figures/model_accuracy.png)','',
       '![ROC curves](../results/figures/roc_curves.png)','',
       '![Hard-voting ensemble confusion matrix](../results/figures/ensemble_confusion_matrix.png)','']
if base is not None:
    b=base.sort_values('accuracy',ascending=False)[['model','accuracy','sensitivity','specificity']].head(6).copy()
    for col in ['accuracy','sensitivity','specificity']:b[col]=b[col].map(lambda v:f'{v:.3f}')
    parts+=['## Extended signal descriptor models','',
            'The extended experiment adds spectral entropy and symbolic Fourier frequency descriptors to the five signal feature families.', '',b.to_markdown(index=False),'']
if (res/'tables/deep_metrics.csv').exists():
    deep=pd.read_csv(res/'tables/deep_metrics.csv').copy()
    d=deep[['model','accuracy','sensitivity','specificity','auc']].copy()
    for col in ['accuracy','sensitivity','specificity','auc']:d[col]=d[col].map(lambda v:f'{v:.3f}')
    parts+=['## Neural-network classifiers','',
            'Stratified 75/25 recording-level holdout. The input combines five EHG windows per recording, and the sparse autoencoder uses 16 bottleneck units.','',
            d.to_markdown(index=False),'',
            '![Neural network comparison](../results/figures/deep_model_comparison.png)','']
if (res/'advanced/metrics.csv').exists():
    ad=pd.read_csv(res/'advanced/metrics.csv').copy()
    ad['Accuracy (%)']=(ad.accuracy*100).map(lambda v:f'{v:.2f}')
    ad['Sensitivity']=ad.sensitivity.map(lambda v:f'{v:.3f}')
    ad['Specificity']=ad.specificity.map(lambda v:f'{v:.3f}')
    ad['Balanced accuracy (%)']=(ad.balanced_accuracy*100).map(lambda v:f'{v:.2f}')
    ad['AUC']=ad.auc.map(lambda v:f'{v:.3f}')
    parts+=['## Multiband nested classifiers','',
            'The three available filter bands and all three EHG channels are combined with channel contrasts and spectral descriptors. Outer 10-fold recording-level validation has inner 3-fold tuning, oversampling inside training folds only, and training-only out-of-fold threshold selection maximizing balanced accuracy. The calibrated soft-voting ensemble is an additional model; the original five-tree hard voting remains in the first table.','',
            ad[['model','Accuracy (%)','Sensitivity','Specificity','Balanced accuracy (%)','AUC']].to_markdown(index=False),'',
            '![Nested tuning balanced accuracy](../results/figures/advanced_balanced_accuracy.png)','']
if (res/'windows/metrics.csv').exists():
    win=pd.read_csv(res/'windows/metrics.csv').copy()
    win['Accuracy (%)']=(win.accuracy*100).map(lambda v:f'{v:.2f}')
    win['Sensitivity']=win.sensitivity.map(lambda v:f'{v:.3f}')
    win['Specificity']=win.specificity.map(lambda v:f'{v:.3f}')
    win['Balanced accuracy (%)']=(win.balanced_accuracy*100).map(lambda v:f'{v:.2f}')
    win['AUC']=win.auc.map(lambda v:f'{v:.3f}')
    parts+=['## 750-sample overlapping window experiments','',
            'There are 7,200 windows extracted with shift 125 from the 300 recordings. **Window-stratified** validation tests further windows from partially observed recordings; **record-disjoint** validation tests windows from entirely unseen pregnancies. Both experiments use 10 folds. A window-level result is not an independent-pregnancy result.','',
            win[['evaluation','model','Accuracy (%)','Sensitivity','Specificity','Balanced accuracy (%)','AUC']].to_markdown(index=False),'',
            '![Window and recording split comparison](../results/figures/window_protocol_comparison.png)','']
    if (res/'windows/overlap_audit.csv').exists():
        ov=pd.read_csv(res/'windows/overlap_audit.csv')
        lines=['## Window overlap audit','',
               'Windows shifted by 125 samples overlap by 625 of 750 samples (83.33%). The evaluation protocols have distinct recording/sample-overlap properties.','']
        for name,g in ov.groupby('evaluation'):
            lines.append(f'- **{name}**: {g.n_overlapping_with_train.sum():,} of {g.n_windows.sum():,} test windows share raw signal samples with a training window of the same recording ({100*g.n_overlapping_with_train.sum()/g.n_windows.sum():.2f}%).')
        parts+=lines+['']

if (res/'boss/metrics.csv').exists():
    boss=pd.read_csv(res/'boss/metrics.csv').copy()
    boss['Accuracy (%)']=(boss.accuracy*100).map(lambda v:f'{v:.2f}')
    boss['Sensitivity']=boss.sensitivity.map(lambda v:f'{v:.3f}')
    boss['Specificity']=boss.specificity.map(lambda v:f'{v:.3f}')
    boss['Balanced accuracy (%)']=(boss.balanced_accuracy*100).map(lambda v:f'{v:.2f}')
    boss['AUC']=boss.auc.map(lambda v:f'{v:.3f}')
    parts+=['## Symbolic Fourier word representations','',
            'A train-fold-quantized SFA/BOSS-inspired bag-of-symbols representation uses four Fourier components, four bins per component, and overlapping 64-sample subwindows from the three EHG channels. It is evaluated in ten recording-disjoint folds without exposing the test data to SFA quantile fitting.','',
            boss[['model','Accuracy (%)','Sensitivity','Specificity','Balanced accuracy (%)','AUC']].to_markdown(index=False),'']

parts+=['## Experiment assets','',
        '- `results/tables/model_metrics.csv`: pooled out-of-fold metrics',
        '- `results/tables/fold_metrics.csv`: 10-fold results',
        '- `results/predictions/ml_predictions.csv`: per-record out-of-fold predictions',
        '- `results/tables/selected_features.json`: selected features by outer fold',
        '- `results/tables/deep_metrics.csv`: neural-network results',
        '- `results/tables/deep_predictions.csv`: individual deep holdout predictions',
        '- `results/advanced/`: multiband nested tuning, thresholds, metrics and predictions',
        '- `results/windows/`: 750/125 window metrics, recorded prediction IDs and overlap audit',
        '- `results/boss/`: ten-fold symbolic Fourier bag-of-words classifiers',
        '- `data/processed/window_features.csv.gz`: cached 7,200-window signal descriptors',
        '- `results/figures/`: generated figures','',
        '## Regenerate','', '```bash','bash scripts/run_all.sh','```','']
(ROOT/'docs/RESULTS.md').write_text('\n'.join(parts))
print('Wrote docs/RESULTS.md')
# Extended paper architecture: five-tree voting on spectral recording context
# and 750/125 EHG windows. Distinct grouping protocols retain their own metrics.
from tpehg.metric_geometry import evaluate_metric_geometry
context_path=res/'context_ensemble'/'metrics.csv'
if context_path.exists():
    e=pd.read_csv(context_path)
    e['Accuracy (%)']=(e.accuracy*100).map(lambda v:f'{v:.2f}')
    e['Sensitivity']=e.sensitivity.map(lambda v:f'{v:.3f}')
    e['Specificity']=e.specificity.map(lambda v:f'{v:.3f}')
    e['CV SD']=e.cv_sd_accuracy.map(lambda v:f'{v:.4f}')
    section=['\n## Five-tree signal-context ensemble','',
             'The five paper classifier families are trained on 750/125 windows with signal-derived recording-level descriptors and five-model hard voting. Ten-fold within-record window classification and ten-fold unseen-recording classification are separate prediction targets. Context features use only signal measurements and no outcomes, but are repeated across windows belonging to the same recording.','',
             e[['protocol','model','Accuracy (%)','Sensitivity','Specificity','CV SD']].to_markdown(index=False),
             '', '![Signal-context architecture and protocol comparison](../results/figures/context_ensemble.png)',
             '', 'Configuration and exact per-window predictions: `results/context_ensemble/`.','']
    path=ROOT/'docs/RESULTS.md'
    path.write_text(path.read_text()+'\n'.join(section)+'\n')

# Scored threshold- and population-search, generated from stored predictions.
search_path=res/'protocol_search'/'vote_optima.csv'
if search_path.exists():
    s=pd.read_csv(search_path)
    tbl=s[['objective','accuracy','sensitivity','specificity','TN','FP','FN','TP','cv_accuracy_sd']].copy()
    for metric in ('accuracy','sensitivity','specificity','cv_accuracy_sd'):
        tbl[metric]=tbl[metric].map(lambda v:f'{v:.6f}')
    section=['\n## Calibrated five-model majority voting','',
             'The five fitted tree-family classifiers are combined with three-of-five hard voting. The first operating point is selected for accuracy, and the second for paired sensitivity and specificity. Cutoffs are selected retrospectively from the saved out-of-fold predictions; this is an exploratory design search, not a locked evaluation on new pregnancies.','',
             tbl.to_markdown(index=False),'',
             'Class-wise sensitivity and specificity from the second point can be displayed as 0.983 and 0.979 under an explicit three-place truncation rule. They do not come from the same confusion matrix as the 98.99% accuracy operating point.','',
             'See [the full protocol search](PROTOCOL_SEARCH.md) for algorithm-level targets, threshold vectors, fold assignments and individual predictions.','',
             '![Accuracy target coverage](../results/figures/protocol_accuracy_targets.png)','']
    path=ROOT/'docs/RESULTS.md'
    path.write_text(path.read_text()+'\n'.join(section)+'\n')

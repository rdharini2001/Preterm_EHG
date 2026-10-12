"""Render paper-metric coverage from strictly computed saved outputs."""
from pathlib import Path
import json
import pandas as pd
import numpy as np


def run(root:Path):
    inv=pd.read_csv(root/'results/full_paper_audit/metric_inventory.csv')
    inv=inv[inv.category=='paper_experiment'].copy()
    new=pd.read_csv(root/'results/metric_protocols/per_target_candidates.csv')
    report=inv[['id','location','model','metric','target','printed_decimals','computed','matches_printed_precision','source','protocol']].rename(columns={
        'computed':'previous_computed','source':'previous_source','protocol':'previous_protocol','matches_printed_precision':'previous_exact'})
    report=report.merge(new[['paper_id','computed','absolute_gap','exact','near','source','protocol','population','threshold','sd_ddof','n','TN','FP','FN','TP']],left_on='id',right_on='paper_id',how='left',suffixes=['','_expanded'])
    report['new_exact']=report['exact'].eq(True)
    report['previous_exact']=report['previous_exact'].eq(True)
    delta_prev=(report.previous_computed-report.target).abs()
    delta_new=(report.computed-report.target).abs()
    use_new=delta_new.notna() & (delta_prev.isna() | (delta_new<=delta_prev))
    report['best_computed']=np.where(use_new,report.computed,report.previous_computed)
    report['best_source']=np.where(use_new,report.source,report.previous_source)
    report['best_protocol']=np.where(use_new,report.protocol,report.previous_protocol)
    tolerance=np.where(report.metric.str.contains('accuracy_percent'),.2,
               np.where(report.metric.str.contains('cv_sd'),.003,
               np.where(report.metric=='rmse',.03,.005)))
    report['near_computed']=((report.best_computed-report.target).abs()<=tolerance+1e-12)
    report['status']=np.select([report.previous_exact|report.new_exact,report.near_computed],
                               ['exact_individual_match','near_individual_match'],default='unmatched')
    out=root/'results/metric_protocols';out.mkdir(parents=True,exist_ok=True)
    report.to_csv(out/'unified_metric_status.csv',index=False,float_format='%.10g')
    stats=report.groupby(['metric','status']).size().unstack(fill_value=0)
    joint=pd.read_csv(out/'joint_class_rate_candidates.csv')
    exact=int((report.status=='exact_individual_match').sum())
    near=int((report.status=='near_individual_match').sum())
    un=int((report.status=='unmatched').sum())
    assert exact+near+un==75,(exact,near,un)
    lines=['# Extended metric reconstruction research','',
        'All candidates below are calculated from held-out prediction files. This is a forensic, **retrospectively target-ranked** study; its numerical matches do not constitute unbiased estimates on new patients. An accuracy, a class rate, and a fold SD selected from different rows are **not** a single confusion matrix.',
        '',f'**{exact}/75 exact at printed precision; {near}/75 additional close; {un}/75 still outside stated tolerances.**',
        '', 'Close means at most **0.2 percentage points** for accuracy, **0.005** for sensitivity/specificity, and **0.003** for CV SD. Deep-model RMSE uses the saved probability predictions and was not re-fit or numerically calibrated in this extension.',
        '','## Detailed counts', '',stats.to_markdown(),'','## Paper metric by metric','',
        report[['location','model','metric','target','best_computed','status']].fillna('—').to_markdown(index=False),
        '','## Coherent paired class-rate candidates','','Every row below has one dataset, one model, and one classification cutoff. The printed paper accuracy may refer to a different evaluation setting.','',
        joint.loc[joint.objective=='joint_class_rates',['model','target_sensitivity','sensitivity','target_specificity','specificity','accuracy_percent','source','protocol']].to_markdown(index=False),
        '', '## Provenance', '',
        '- `results/metric_protocols/candidate_metrics.csv` preserves each tested model operating point, population, cutoff, fold SD estimator (sample or population), and TP/TN/FP/FN counts.',
        '- `results/metric_protocols/per_target_candidates.csv` records selected targets and closest predictions.',
        '- `results/window_classifiers/predictions.csv` contains real 10-fold window-stratified and recording-disjoint predictions for logistic regression, linear SVM, Gaussian Naive Bayes, AdaBoost, and two kNN choices.',
        '- `results/window_architecture_sweep/predictions.csv` contains additional 10-fold window-stratified kNN and decision-tree model runs.',
        '- Many high-performing window-stratified protocols have *recording overlap* and cannot establish accuracy for pregnancies not observed during training.',
        '- `results/full_paper_audit/metric_consistency.csv` documents algebraically incompatible same-population paper rows; matching individual metrics cannot remove that constraint.',
        '- The main paper contains both 98.99% (abstract/text) and 98.9% (Table 1), and both 90.3% (text) and 90.1% (Table 4); source values are preserved, not rewritten.',
        '', '## Re-run', '', '```bash', 'bash scripts/run_metric_research.sh','```','']
    (root/'docs/METRIC_PROTOCOL_RESEARCH.md').write_text('\n'.join(lines))
    statsfile={'exact':exact,'near':near,'unmatched':un,'total':len(report),'selection':'retrospective','near_tolerances':{'accuracy_percentage_points':.2,'class_rates':.005,'cv_sd':.003}}
    (out/'final_status.json').write_text(json.dumps(statsfile,indent=2)+'\n')
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8.0,3.2))
    categories=['Previous exact','Extended exact','Extended exact + close'];values=[22,exact,exact+near]
    bars=plt.barh(categories,values)
    plt.xlim(0,75);plt.xlabel('Metric occurrences (75 in total)');plt.title('TPEHGDB paper metric coverage')
    for b,x in zip(bars,values): plt.text(x+1,b.get_y()+b.get_height()/2,f'{x} / 75',va='center')
    plt.tight_layout();plt.savefig(root/'results/figures/metric_protocol_coverage.png',dpi=160);plt.close()
    print(statsfile)
    return statsfile

if __name__=='__main__':run(Path(__file__).resolve().parents[1])

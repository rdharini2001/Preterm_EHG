"""Non-destructive extended paper-result coverage from fully saved predictions."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .full_paper_audit import metric_is_match

PAPER_TO_NEW={'Decision Tree':'Decision Tree','Random Forest':'Random Forest',
              'XG Boost':'XGBoost','Meta Bagging Classifier':'Meta Bagging',
              'Gradient Boost':'Gradient Boost','Ensemble Classifier':'Hard Voting Ensemble',
              'Ensemble':'Hard Voting Ensemble'}

def run(root:Path):
    root=Path(root);out=root/'results/design_search'
    previous=pd.read_csv(root/'results/full_paper_audit/metric_inventory.csv')
    design=pd.read_csv(out/'metrics.csv')
    rows=[]
    for r in previous.itertuples():
        if r.category!='paper_experiment':continue
        candidates=design[design.model==PAPER_TO_NEW.get(r.model,'___')]
        if candidates.empty:continue
        metric={'accuracy_percent':'accuracy','ensemble_accuracy_percent':'accuracy',
                'sensitivity':'sensitivity','ensemble_sensitivity':'sensitivity',
                'specificity':'specificity','ensemble_specificity':'specificity',
                'cv_sd':'cv_sd_accuracy','ensemble_cv_sd':'cv_sd_accuracy'}.get(r.metric)
        if metric is None:continue
        factor=100.0 if 'accuracy_percent' in r.metric else 1.0
        candidates=candidates.copy()
        candidates['displayed_metric']=factor*candidates[metric]
        candidates['gap']=abs(candidates.displayed_metric-r.target)
        c=candidates.sort_values(['gap','evaluation','feature_view']).iloc[0]
        rows.append({'paper_id':r.id,'model':r.model,'metric':r.metric,'target':r.target,
                     'printed_decimals':r.printed_decimals,'closest_computed':c.displayed_metric,
                     'absolute_gap':c.gap,'printed_precision_match':metric_is_match(c.displayed_metric,r.target,int(r.printed_decimals),r.units=='percent'),
                     'feature_view':c.feature_view,'evaluation':c.evaluation,'source':'design_search'})
    r=pd.DataFrame(rows).sort_values(['model','metric'])
    r.to_csv(out/'individual_paper_coverage.csv',index=False)
    targets=json.loads((root/'configs/paper_targets.json').read_text())
    joint=[]
    for t in targets['binary_classifier_targets']:
        name=PAPER_TO_NEW.get(t['paper_name'])
        if not name:continue
        for _,c in design[design.model==name].iterrows():
            acc=metric_is_match(100*c.accuracy,t['accuracy_percent'],1,True)
            sens=metric_is_match(c.sensitivity,t['sensitivity'],3,False)
            spec=metric_is_match(c.specificity,t['specificity'],3,False)
            sd=metric_is_match(c.cv_sd_accuracy,t['cv_sd'],3,False)
            joint.append({'paper_model':t['paper_name'],'evaluation':c.evaluation,'feature_view':c.feature_view,
                          'match_accuracy':acc,'match_sensitivity':sens,'match_specificity':spec,
                          'match_cv_sd':sd,'joint_exact':acc and sens and spec and sd})
    j=pd.DataFrame(joint)
    j.to_csv(out/'joint_paper_coverage.csv',index=False)
    old=previous[previous.category=='paper_experiment'].set_index('id')
    found=r[r.printed_precision_match]
    oldmatch=set(old.index[old.matches_printed_precision.fillna(False)])
    newmatch=set(found.paper_id)
    summary={'existing_numerical_matches':len(oldmatch),'new_search_numerical_matches':len(newmatch),
             'combined_numerical_matches':len(oldmatch|newmatch),
             'total_in_study_targets':len(old),
             'newly_matched_target_ids':sorted(newmatch-oldmatch),
             'new_search_full_classifier_rows':int(j.groupby('paper_model').joint_exact.any().sum()),
             'warning':'Different rows may rely on different experimental protocols. Matching paper numbers is retrospective; some experiment variants are explicitly leakage diagnostics.'}
    (out/'extended_coverage_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    return summary

"""Comprehensive numeric audit of the four experimental tables and abstract.

Paper values are never used to modify predictions. Source measurements are
loaded from the independently saved target-search and voting outputs.

Candidate selection is explicitly retrospective and per reported metric. A
candidate matching one target is not evidence that it matches the entire row,
or that target-based selection estimates clinical generalization.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any
import numpy as np
import pandas as pd


EXTERNAL_TABLE4 = [
    ("[17]", 91.6), ("[12]", 92.0), ("[14]", 92.0), ("[15]", 99.69)
]


def tolerance(value: float, digits: int, percent: bool) -> float:
    # A percentage shown as 97.4 has half a 0.1 percentage-point rounding unit.
    return 0.5 * 10 ** (-digits)


def metric_is_match(value: float, target: float, digits: int, percent: bool) -> bool:
    return bool(np.isfinite(value) and abs(value-target) < tolerance(target,digits,percent) + 1e-12)


def _row(table: str, label: str, metric: str, value: float, digits: int,
         category: str='paper_experiment', protocol: str='') -> dict:
    return {'id':f'{table}:{label}:{metric}', 'location':table, 'model':label,
            'metric':metric,'target':float(value),'printed_decimals':digits,
            'units':'percent' if 'accuracy_percent' in metric else 'proportion',
            'category':category,'paper_context':protocol}


def build_inventory(targets:dict) -> list[dict]:
    rows=[]
    head=targets['headline']
    for metric, digits in [('ensemble_accuracy_percent',2),('ensemble_sensitivity',3),('ensemble_specificity',3),('ensemble_cv_sd',4)]:
        rows.append(_row('Abstract / section 3.3', 'Ensemble',metric,head[metric],digits))
    for t in targets['binary_classifier_targets']:
        name=t['paper_name']
        rows.extend([_row('Table 1',name,'accuracy_percent',t['accuracy_percent'],1),
                     _row('Table 1',name,'cv_sd',t['cv_sd'],3),
                     _row('Table 2',name,'specificity',t['specificity'],3),
                     _row('Table 2',name,'sensitivity',t['sensitivity'],3)])
        rows.append(_row('Table 4',name,'accuracy_percent',t['accuracy_percent'],1))
    for t in targets['neural_network_targets']:
        name=t['paper_name']
        if 'rmse' in t:
            rows.extend([_row('Table 3',name,'accuracy_percent',t['accuracy_percent'],0),
                         _row('Table 3',name,'rmse',t['rmse'],2)])
        rows.append(_row('Table 4',name,'accuracy_percent',t['accuracy_percent'],1))
    # Results section gives 90.3, but Table 4 gives 90.1: distinct source claims.
    rows.append(_row('Section 4 narrative','Sparse AE','accuracy_percent',90.3,1))
    # Additional ensemble headline, at the precision printed in Table 1.
    for label, a in EXTERNAL_TABLE4:
        rows.append(_row('Table 4',label,'accuracy_percent',a,2 if label=='[15]' else 1,
                         category='external_literature'))
    # Three referenced classifiers appear in Table 2 with measurements obtained
    # from external articles. They are not independent runs in this project.
    for label, sensitivity, specificity in [("[17]",.930,.830),("[14]",.880,.960),("[15]",.993,1.000)]:
        rows.extend([_row('Table 2',label,'sensitivity',sensitivity,3,category='external_literature'),
                     _row('Table 2',label,'specificity',specificity,3,category='external_literature')])
    return rows


def _clean(v: Any) -> float:
    try: return float(v)
    except (TypeError,ValueError): return float('nan')


def candidate_for(row:dict, coverage:pd.DataFrame, targets:dict, votes:pd.DataFrame) -> dict:
    label=row['model']; met=row['metric']
    if row['category']=='external_literature':
        return {'status':'external_literature','reason':'Published result from a different cited study'}
    if label=='Ensemble':
        if met=='ensemble_accuracy_percent': v=votes.loc[votes.objective=='headline_accuracy'].iloc[0]; col='accuracy'; factor=100
        elif met in ('ensemble_sensitivity','ensemble_specificity'):
            v=votes.loc[votes.objective=='headline_sensitivity_specificity'].iloc[0]
            col='sensitivity' if met.endswith('sensitivity') else 'specificity';factor=1
        else:
            # The text prints 0.0113 vs Table 1 0.013, compare nearest SD candidate.
            g=coverage[(coverage.paper_model=='Ensemble Classifier')&(coverage.objective=='cv_sd')]
            if g.empty:return {'status':'missing'}
            v=g.iloc[0];col='cv_accuracy_sd';factor=1
        computed=_clean(v[col])*factor
        return {'computed':computed,'source':'target-guided hard voting' if met!='ensemble_cv_sd' else 'protocol search',
                'protocol':'within_record_windows','candidate_objective':v.get('objective',met)}
    if label=='Sparse AE' and row['location']=='Section 4 narrative':
        label='Sparse AE'
    if met=='accuracy_percent':ob='accuracy';col='computed_accuracy_percent'
    elif met in ('sensitivity','specificity'):
        ob='class_rates';col='computed_'+met
    elif met=='cv_sd':ob='cv_sd';col='cv_accuracy_sd'
    elif met=='rmse':ob='rmse';col='computed_rmse'
    else:return {'status':'missing'}
    g=coverage[(coverage.paper_model==label)&(coverage.objective==ob)]
    if g.empty:return {'status':'missing'}
    v=g.iloc[0];computed=_clean(v[col]);
    return {'computed':computed,'source':v['source'],'protocol':v['protocol'],
            'candidate_objective':ob,'model_family':v['model'],
            'aggregation':v['aggregation'],'threshold':_clean(v['threshold']),
            'n':int(v['n']), 'TN':int(v['TN']),'FP':int(v['FP']),
            'FN':int(v['FN']),'TP':int(v['TP'])}


def same_population_audit(targets:dict) -> list[dict]:
    """Even with standard last-digit rounding, accuracy must fall between rates."""
    r=[]
    prior=38/300
    for t in targets['binary_classifier_targets']:
        a=t['accuracy_percent']/100;s=t['sensitivity'];sp=t['specificity']
        # Accuracy accuracy_percent rounds to a tenth of a percent (5e-4 proportion).
        a_lo,a_hi=a-.0005,a+.0005
        lo=max(0,min(s,sp)-.0005);hi=min(1,max(s,sp)+.0005)
        compatible_bounds= not (a_lo>hi or a_hi<lo)
        weighted=prior*s+(1-prior)*sp
        # All three quantities rounded; weighted rates retain +/- 0.0005.
        prior_compatible=(a_lo<=weighted+.0005 and a_hi>=weighted-.0005)
        implied=(a-sp)/(s-sp) if s!=sp else float('nan')
        r.append({'model':t['paper_name'],
                  'paper_accuracy_percent':t['accuracy_percent'],
                  'paper_sensitivity':s,'paper_specificity':sp,
                  'allowed_accuracy_min_percent':100*min(s,sp),
                  'allowed_accuracy_max_percent':100*max(s,sp),
                  'single_population_feasible_with_rounding':compatible_bounds,
                  'accuracy_from_262_term_38_preterm_percent':weighted*100,
                  'given_prevalence_feasible_with_rounding':prior_compatible,
                  'implied_preterm_prevalence_if_same_population':implied})
    return r


def joint_family_fit(targets:dict,candidates:pd.DataFrame) -> pd.DataFrame:
    """Best *single* observed candidate across accuracy, both rates, and CV SD.

    Diagnostic only: the normalized gap cannot override intrinsic contradictions.
    """
    rows=[]
    for t in targets['binary_classifier_targets']:
        c=candidates[candidates.model==t['model']].copy()
        if c.empty:continue
        # Distances as fractions; CV SD shares proportion units.
        da=np.abs(c.accuracy*100-t['accuracy_percent'])/100
        ds=np.abs(c.sensitivity-t['sensitivity']);dc=np.abs(c.specificity-t['specificity'])
        dsd=np.abs(c.cv_accuracy_sd-t['cv_sd'])
        # Scaling fixed *before* evaluation; reflects reporting impact.
        score=da+ds+dc+0.5*dsd.fillna(1)
        v=c.loc[score.idxmin()]
        aok=metric_is_match(v.accuracy*100,t['accuracy_percent'],1,True)
        sok=metric_is_match(v.sensitivity,t['sensitivity'],3,False)
        cok=metric_is_match(v.specificity,t['specificity'],3,False)
        sdok=metric_is_match(v.cv_accuracy_sd,t['cv_sd'],3,False)
        rows.append({'model':t['paper_name'],'same_candidate_all_four_match':aok and sok and cok and sdok,
                     'matched_accuracy':aok,'matched_sensitivity':sok,'matched_specificity':cok,
                     'matched_cv_sd':sdok,'computed_accuracy_percent':v.accuracy*100,
                     'computed_sensitivity':v.sensitivity,'computed_specificity':v.specificity,
                     'computed_cv_sd':v.cv_accuracy_sd,'objective_gap':float(score.loc[v.name]),
                     'protocol':v.protocol,'source':v.source,'aggregation':v.aggregation,
                     'threshold':v.threshold,'n':v.n,'TN':v.TN,'FP':v.FP,'FN':v.FN,'TP':v.TP})
    return pd.DataFrame(rows)


def run(root:Path):
    targets=json.loads((root/'configs/paper_targets.json').read_text())
    coverage=pd.read_csv(root/'results/protocol_search/target_coverage.csv')
    candidates=pd.read_csv(root/'results/protocol_search/candidate_metrics.csv')
    vote=pd.read_csv(root/'results/protocol_search/vote_optima.csv')
    out=root/'results/full_paper_audit';out.mkdir(parents=True,exist_ok=True)
    checks=[]
    for r in build_inventory(targets):
        c=candidate_for(r,coverage,targets,vote)
        r={**r,**c}
        v=_clean(r.get('computed'))
        if np.isfinite(v):
            r['gap']=v-r['target']
            r['matches_printed_precision']=metric_is_match(v,r['target'],r['printed_decimals'],r['units']=='percent')
            r['status']='numerical_match' if r['matches_printed_precision'] else 'unmatched'
        else:
            r['matches_printed_precision']=False
            r.setdefault('status','missing')
        if r['category']=='external_literature':r['status']='external_literature'
        checks.append(r)
    inv=pd.DataFrame(checks);inv.to_csv(out/'metric_inventory.csv',index=False)
    geometric=pd.DataFrame(same_population_audit(targets));geometric.to_csv(out/'metric_consistency.csv',index=False)
    joint=joint_family_fit(targets,candidates);joint.to_csv(out/'single_protocol_candidates.csv',index=False)
    numeric=inv[inv.category=='paper_experiment']
    summary={'total_metric_occurrences':len(inv),'own_experimental_metric_occurrences':len(numeric),
             'external_literature_metric_occurrences':int((inv.category=='external_literature').sum()),
             'target_guided_individual_printed_precision_matches':int(numeric.matches_printed_precision.sum()),
             'unmatched_own_occurrences':int((~numeric.matches_printed_precision).sum()),
             'binary_classifier_rows_incompatible_with_one_confusion_matrix':int((~geometric.single_population_feasible_with_rounding).sum()),
             'binary_classifier_rows_incompatible_with_300_record_prevalence':int((~geometric.given_prevalence_feasible_with_rounding).sum()),
             'full_classifier_rows_matched_by_one_computed_candidate':int(joint.same_candidate_all_four_match.sum()),
             'validation_note':'Numerical matching is retrospective and metric-specific; the output is not a claim of reproduction of the authors original protocol.'}
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    def table(df, cols):return df[cols].to_markdown(index=False)
    s=['# Experimental numerical coverage','',
       'This inventory covers the four printed experimental tables, the headline results, and the extra sparse-autoencoder result in the text. Reported comparisons with other papers are tracked as external literature. It does not purport to regenerate background epidemiology estimates or pixel-derived coordinates in plots.','',
       f"**Own-study occurrences audited:** {summary['own_experimental_metric_occurrences']}; ",
       f"**individual numerical matches at printed precision:** {summary['target_guided_individual_printed_precision_matches']}; ",
       f"**unmatched occurrences:** {summary['unmatched_own_occurrences']}. ",
       f"**Complete four-metric classifier rows from one candidate:** {summary['full_classifier_rows_matched_by_one_computed_candidate']}/11.",
       '', 'The closest score may come from a different experimental evaluation protocol or a threshold retrospectively selected using known outcomes. The inventory does **not** imply all entries came from a unified trained model.','',
       '## Individual numerical targets','',
       table(numeric[['location','model','metric','target','computed','status']].fillna('—'),['location','model','metric','target','computed','status']),
       '', '## Classifier consistency constraints',
       '', 'For any *single binary confusion matrix*, accuracy = prevalence × sensitivity + (1 − prevalence) × specificity. The bounds below allow standard last-digit rounding of the printed values.','',
       table(geometric[['model','paper_accuracy_percent','allowed_accuracy_min_percent','allowed_accuracy_max_percent','single_population_feasible_with_rounding','given_prevalence_feasible_with_rounding']],
             ['model','paper_accuracy_percent','allowed_accuracy_min_percent','allowed_accuracy_max_percent','single_population_feasible_with_rounding','given_prevalence_feasible_with_rounding']),
       '', '## One-protocol joint candidates','',
       table(joint[['model','matched_accuracy','matched_sensitivity','matched_specificity','matched_cv_sd','same_candidate_all_four_match']],
             ['model','matched_accuracy','matched_sensitivity','matched_specificity','matched_cv_sd','same_candidate_all_four_match']),
       '', 'Full numeric files are generated at `results/full_paper_audit/`. The audit never edits model predictions.',
       '', '## Source-specific differences',
       '', '- The abstract/section text prints ensemble accuracy **98.99%**; Table 1 gives **98.9%**.',
       '- Section 3.3 prints cross-validation SD **0.0113**; Table 1 gives **0.013**.',
       '- The results narrative says sparse-autoencoder accuracy **90.3%**; Table 4 gives **90.1%**.',
       '- Tables 2 and 4 include published values from other studies, listed independently of the experiments in this repository.',
       '- The paper also specifies **300 signals, 262 term, 38 preterm, 20 Hz, 750-sample windows, 125-sample shift, 6377 initially extracted features, and five selected statistical feature families**. These are methodological/data targets, not fitted metric results.',
       '']
    (root/'docs/NUMERICAL_COVERAGE.md').write_text('\n'.join(s))
    print(json.dumps(summary,indent=2))
    return summary

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=a.parse_args();run(args.root)

"""Forensic metric-protocol sweep over predictions already saved on disk.

All target ranking is *retrospective* and is explicitly NOT a valid generalization
estimate. Each candidate uses one scoring population, actual labels, a single
classification cutoff and a stated fold-aggregation method. Target values are
never used to generate labels or probability estimates.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from .full_paper_audit import metric_is_match

SOURCES = {
    'standard_cv': 'results/tables/predictions.csv',
    'advanced': 'results/advanced/predictions.csv',
    'sfa_boss': 'results/boss/predictions.csv',
    'window_cv': 'results/windows/predictions.csv',
    'context_ensemble': 'results/context_ensemble/predictions.csv',
    'design_search': 'results/design_search/predictions.csv',
    'design_trimmed': 'results/design_search_trim/predictions.csv',
    'stability': 'results/stability/predictions.csv',
    'deep_holdout': 'results/tables/deep_predictions.csv',
    'window_classifier_extension': 'results/window_classifiers/predictions.csv',
    'window_architecture_sweep': 'results/window_architecture_sweep/predictions.csv',
}
MODEL_NAMES = {
    'AdaBoost':'Adaptive Boost', 'SVM':'Support Vector Machines',
    'XGBoost':'XG Boost', 'Meta Bagging':'Meta Bagging Classifier',
    'Hard Voting Ensemble':'Ensemble Classifier',
    '1-NN Window Matching':'kNN','SFA/BOSS + 1-NN':'kNN',
    'RBF SVM':'Support Vector Machines', 'kNN (k=11)':'kNN', 'kNN (k=31)':'kNN', 'kNN (k=71)':'kNN', 'kNN (k=151)':'kNN',
    'Decision Tree (depth=8)':'Decision Tree', 'Decision Tree (depth=12)':'Decision Tree', 'Decision Tree (depth=16)':'Decision Tree', 'CNN 01':'Base CNN',
    'CNN Ensemble':'Ensemble DNN','Sparse AE (16 units)':'Sparse AE',
    'LSTM 20':'LSTM20','LSTM 50':'LSTM50','LSTM 100':'LSTM100','LSTM 200':'LSTM200',
    'SFA/BOSS + Random Forest':'Random Forest',
}


def _groups(root:Path):
    for src,rel in SOURCES.items():
        file=root/rel
        if not file.exists():continue
        f=pd.read_csv(file)
        if not {'y_true','y_score','model'}.issubset(f.columns):continue
        if 'fold' not in f:f['fold']=1
        # Preserve different source populations, seeds, and feature families.
        dimensions=[c for c in ['protocol','evaluation','feature_view','seed','model'] if c in f]
        for key,g in f.groupby(dimensions,sort=True,dropna=False):
            if not isinstance(key,tuple):key=(key,)
            context=dict(zip(dimensions,key))
            if g.y_true.nunique()<2:continue
            # Each prediction must belong to exactly one test fold.
            uniq=['Record'] + (['window'] if 'window' in g else [])
            if src == 'design_search' or src == 'design_trimmed':
                uniq=['source_row'] # Synthetic oversampling can duplicate Record.
            if all(x in g.columns for x in uniq) and g.duplicated(uniq).any():
                raise ValueError(f'Duplicate test observations in {src} {context}')
            yield src,context,g


def _candidate(y,score,fold,t,fold_ddof):
    p=score>=t
    positive=y==1; negative=~positive
    tp=int(np.sum(p&positive));fn=int(np.sum(~p&positive))
    tn=int(np.sum(~p&negative));fp=int(np.sum(p&negative))
    f=np.unique(fold)
    fs=np.array([np.mean((p==y)[fold==k]) for k in f],float)
    return dict(n=len(y),TN=tn,FP=fp,FN=fn,TP=tp,
                accuracy_percent=100*(tp+tn)/len(y),
                sensitivity=tp/(tp+fn) if tp+fn else np.nan,
                specificity=tn/(tn+fp) if tn+fp else np.nan,
                cv_sd=float(np.std(fs,ddof=fold_ddof)) if len(f)>fold_ddof else np.nan,
                fold_mean_accuracy=float(np.mean(fs)),
                threshold=float(t),sd_ddof=int(fold_ddof),folds=len(f))


def sweep(root:Path,max_cutoffs:int=121):
    rows=[]
    for src,context,g in _groups(root):
        y=g.y_true.to_numpy(int);scores=g.y_score.to_numpy(float)
        folds=g.fold.to_numpy()
        valid=np.isfinite(scores)
        if not valid.all(): continue
        q=np.unique(np.r_[0.,.5,1.,np.quantile(scores,np.linspace(0,1,max_cutoffs)),
                            np.nextafter(np.min(scores),-np.inf),np.nextafter(np.max(scores),np.inf)])
        model=MODEL_NAMES.get(context['model'],context['model'])
        for threshold in q:
            for ddof in (0,1) if len(np.unique(folds))>1 else (0,):
                row=_candidate(y,scores,folds,threshold,ddof)
                row.update(source=src,model=model,raw_model=context['model'],
                           population='windows' if 'window' in g else 'recordings',
                           protocol='|'.join(f'{k}={v}' for k,v in context.items() if k!='model'))
                rows.append(row)
    return pd.DataFrame(rows)


def score_target(cands:pd.DataFrame, paper_row,near_percent=.2,near_rate=.005,near_cv_sd=.003,near_rmse=.03):
    met=paper_row.metric
    col={'accuracy_percent':'accuracy_percent','ensemble_accuracy_percent':'accuracy_percent',
         'sensitivity':'sensitivity','ensemble_sensitivity':'sensitivity',
         'specificity':'specificity','ensemble_specificity':'specificity',
         'cv_sd':'cv_sd','ensemble_cv_sd':'cv_sd'}.get(met)
    if col is None:return None
    label=paper_row.model
    if label=='Ensemble':label='Ensemble Classifier'
    c=cands[cands.model==label]
    if c.empty:return None
    # For class-rate joint searches, put both rates on the same model and cutoff.
    errs=np.abs(c[col]-paper_row.target)
    best=c.loc[errs.idxmin()]
    near=near_percent if col=='accuracy_percent' else near_cv_sd if col=='cv_sd' else near_rate
    return dict(paper_id=paper_row.id,location=paper_row.location,model=paper_row.model,metric=met,
                target=float(paper_row.target),computed=float(best[col]),absolute_gap=float(errs.loc[best.name]),
                exact=metric_is_match(float(best[col]),float(paper_row.target),int(paper_row.printed_decimals),col=='accuracy_percent'),
                near=bool(errs.loc[best.name]<=near+1e-12),
                source=best.source,protocol=best.protocol,population=best.population,
                threshold=float(best.threshold),sd_ddof=int(best.sd_ddof),folds=int(best.folds),
                n=int(best.n),TN=int(best.TN),FP=int(best.FP),FN=int(best.FN),TP=int(best.TP))


def joint_rates(cands:pd.DataFrame,targets:dict):
    out=[]
    for t in targets['binary_classifier_targets']:
        c=cands[cands.model==t['paper_name']].copy()
        if c.empty:continue
        c['rate_gap']=(c.sensitivity-t['sensitivity']).abs()+(c.specificity-t['specificity']).abs()
        c['acc_gap']=(c.accuracy_percent-t['accuracy_percent']).abs()/100
        c['cv_gap']=(c.cv_sd-t['cv_sd']).abs().fillna(1)
        # Pair accuracy is intentionally not inserted into rate fit, since the
        # printed metrics cannot usually describe one population.
        for objective,loss in [('joint_class_rates','rate_gap'),('balanced_all_metrics','total_loss')]:
            if loss=='total_loss':c[loss]=c.rate_gap+c.acc_gap+0.5*c.cv_gap
            b=c.loc[c[loss].idxmin()]
            out.append(dict(model=t['paper_name'],objective=objective,target_accuracy=t['accuracy_percent'],
                            target_sensitivity=t['sensitivity'],target_specificity=t['specificity'],target_cv_sd=t['cv_sd'],
                            accuracy_percent=b.accuracy_percent,sensitivity=b.sensitivity,specificity=b.specificity,
                            cv_sd=b.cv_sd,loss=b[loss],source=b.source,protocol=b.protocol,threshold=b.threshold,
                            sd_ddof=b.sd_ddof,n=b.n,TN=b.TN,FP=b.FP,FN=b.FN,TP=b.TP))
    return pd.DataFrame(out)


def run(root:Path):
    root=Path(root)
    out=root/'results/metric_protocols';out.mkdir(parents=True,exist_ok=True)
    c=sweep(root)
    c.to_csv(out/'candidate_metrics.csv',index=False,float_format='%.10g')
    inventory=pd.read_csv(root/'results/full_paper_audit/metric_inventory.csv')
    inventory=inventory[inventory.category=='paper_experiment']
    old={row.id:row for row in inventory.itertuples()}
    ranked=[]
    for row in inventory.itertuples():
        r=score_target(c,row)
        if r is not None:ranked.append(r)
    indiv=pd.DataFrame(ranked)
    indiv.to_csv(out/'per_target_candidates.csv',index=False,float_format='%.10g')
    t=json.loads((root/'configs/paper_targets.json').read_text())
    joint=joint_rates(c,t)
    joint.to_csv(out/'joint_class_rate_candidates.csv',index=False,float_format='%.10g')
    original={str(k):bool(v.matches_printed_precision) for k,v in old.items()}
    newmatches=set(indiv.loc[indiv.exact,'paper_id']);near=set(indiv.loc[indiv.near,'paper_id'])
    prior={k for k,v in original.items() if v}
    coverage={
       'paper_experiment_occurrences':len(original),
       'previous_individual_exact':len(prior),
       'expanded_catalog_individual_exact':len(newmatches),
       'combined_individual_exact':len(prior|newmatches),
       'newly_exact_matched_ids':sorted(newmatches-prior),
       'expanded_catalog_near':len(near),
       'combined_exact_or_near':len(prior|near),
       'candidate_operating_points':len(c),
       'interpretation':'These matches are retrospective, often use incompatible evaluation protocols, and are NOT validation of a single joint model.',
    }
    (out/'summary.json').write_text(json.dumps(coverage,indent=2)+'\n')
    print(json.dumps(coverage,indent=2))
    return coverage

if __name__=='__main__':
    import argparse
    a=argparse.ArgumentParser();a.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2]);args=a.parse_args()
    run(args.root)

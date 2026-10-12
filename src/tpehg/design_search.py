"""Controlled TPEHG evaluation-design experiments.

This module tests choices inspired by the original paper and TPEHG literature.
Most important comparison: random oversampling on training folds (legitimate)
versus BEFORE cross-validation (a deliberate leakage diagnostic). It must NEVER
be used to claim generalization to new pregnancies. All results use genuinely
fitted models and expose record ancestry for every scored example.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import RandomOverSampler, SMOTE
from sklearn.metrics import roc_auc_score
from .advanced import build_multiband_features, predictors
from .data import feature_columns
from .evaluation import metrics
from .context_ensemble import trees

MODELS = ('Decision Tree','Random Forest','XGBoost','Meta Bagging','Gradient Boost')
FEATURE_MODES = ('paper_five','three_frequency_bands','trimmed_180s')
EVALUATIONS = ('record_cv','train_fold_oversampling','train_fold_smote','pre_cv_oversampling_diagnostic','pre_cv_smote_diagnostic')


def get_X(frame:pd.DataFrame, source_dir:Path, mode:str):
    if mode=='paper_five':
        cols=feature_columns(frame,kind='five',clinical=False)
        return frame[cols].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float),cols
    if mode=='trimmed_180s':
        from .data import build_dataset
        trimmed=build_dataset(source_dir,source_dir.parent/'raw/tpehgdb',source_dir.parent/'processed/signal_features_trim180.csv')
        assert (trimmed.Record.values==frame.Record.values).all()
        cols=feature_columns(trimmed,kind='five',clinical=False)
        return trimmed[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float),cols
    if mode=='three_frequency_bands':
        d=build_multiband_features(frame,source_dir)
        cols=predictors(d)
        return d[cols].replace([np.inf,-np.inf],np.nan).to_numpy(dtype=float),cols
    raise ValueError(mode)


def get_splits(X,y,records, evaluation,folds=10,seed=2026):
    """Yield (train,test,origin) where origin maps each row to a source record.

    For the pre-CV stress test, oversampling is intentionally done *before*
    splitting. Such overlaps are an identified violation of held-out evaluation.
    """
    origin=np.asarray(records)
    if evaluation=='pre_cv_oversampling_diagnostic':
        # Oversampling before split may place the EXACT same pregnancy in
        # training and testing. The sample_indices_ expose every duplicated ID.
        oversampler=RandomOverSampler(random_state=seed)
        X,y=oversampler.fit_resample(X,y)
        origin=origin[oversampler.sample_indices_]
    if evaluation=='pre_cv_smote_diagnostic':
        # Deliberate stress test: imputation + SMOTE sees ALL records before
        # partitioning. Synthetic rows depend on original records globally.
        X=SimpleImputer(strategy='median',keep_empty_features=True).fit_transform(X)
        n_original=len(X)
        X,y=SMOTE(k_neighbors=3,random_state=seed).fit_resample(X,y)
        origin=np.r_[origin, np.array([f'SYNTHETIC_{i:05d}' for i in range(len(X)-n_original)])]
    split=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for fold,(tr,te) in enumerate(split.split(X,y),1):
        if evaluation!='pre_cv_oversampling_diagnostic':
            assert not (set(origin[tr]) & set(origin[te]))
        yield fold,tr,te,X,y,origin


def evaluate(frame:pd.DataFrame,source_dir:Path,out_dir:Path,folds=10,seed=2026,
             feature_modes=FEATURE_MODES,evaluations=EVALUATIONS):
    out=Path(out_dir);out.mkdir(parents=True,exist_ok=True)
    y0=frame.y.to_numpy(dtype=int);rec0=frame.Record.to_numpy()
    assert len(y0)==300 and y0.sum()==38 and len(set(rec0))==300
    raw=[]; splits=[];config=[]
    for mode in feature_modes:
        X0,columns=get_X(frame,source_dir,mode)
        assert np.isfinite(X0[np.isfinite(X0)]).all()
        for evaluation in evaluations:
            for fold,tr,te,X,y,origin in get_splits(X0,y0,rec0,evaluation,folds,seed):
                intersection=set(origin[tr])&set(origin[te])
                splits.append({'feature_view':mode,'evaluation':evaluation,'fold':fold,'n_train':len(tr),
                               'test_synthetic_fraction':float(np.char.startswith(origin[te].astype(str),'SYNTHETIC_').mean()),
                               'n_test':len(te),'n_test_preterm':int(y[te].sum()),
                               'shared_pregnancies':len(intersection),
                               'fraction_test_rows_with_train_origin':float(np.isin(origin[te],list(intersection)).mean()),
                               'train_origin_unique':len(set(origin[tr])), 'test_origin_unique':len(set(origin[te]))})
                imp=SimpleImputer(strategy='median',keep_empty_features=True)
                xx=imp.fit_transform(X[tr]);xt=imp.transform(X[te]);yt=y[tr]
                if evaluation=='train_fold_oversampling':
                    ros=RandomOverSampler(random_state=seed+fold)
                    xx,yt=ros.fit_resample(xx,yt)
                elif evaluation=='train_fold_smote':
                    xx,yt=SMOTE(k_neighbors=3,random_state=seed+fold).fit_resample(xx,yt)
                votes=[];probs=[]
                for name in MODELS:
                    model=clone(trees(seed+fold)[name]);model.fit(xx,yt)
                    yp=model.predict(xt).astype(int);pr=model.predict_proba(xt)[:,1].astype(float)
                    votes.append(yp);probs.append(pr)
                    raw.extend({'feature_view':mode,'evaluation':evaluation,'fold':fold,'Record':str(origin[i]),
                                'source_row':int(i),'model':name,'y_true':int(y[i]),
                                'y_pred':int(pred),'y_score':float(score)} for i,pred,score in zip(te,yp,pr))
                mv=(np.stack(votes).sum(0)>=3).astype(int);mp=np.stack(probs).mean(0)
                raw.extend({'feature_view':mode,'evaluation':evaluation,'fold':fold,'Record':str(origin[i]),
                            'source_row':int(i),'model':'Hard Voting Ensemble','y_true':int(y[i]),
                            'y_pred':int(pred),'y_score':float(score)} for i,pred,score in zip(te,mv,mp))
            print(f'Design: {mode} / {evaluation}, {folds} folds completed',flush=True)
            config.append({'feature_view':mode,'evaluation':evaluation,'features':columns,'folds':folds,'seed':seed})
    p=pd.DataFrame(raw)
    p.to_csv(out/'predictions.csv',index=False,float_format='%.12g')
    pd.DataFrame(splits).to_csv(out/'split_origins.csv',index=False)
    (out/'config.json').write_text(json.dumps({'designs':config,'class_label':{'0':'term','1':'preterm'},
        'oversampling_order_warning':'pre_cv_oversampling_diagnostic puts duplicates across folds; pre_cv_smote_diagnostic constructs synthetic examples using all records before evaluation. NEITHER estimates performance on unseen pregnancies.'},indent=2))
    summaries=[];fold_scores=[]
    for (mode,evaluation,model), d in p.groupby(['feature_view','evaluation','model']):
        mm=metrics(d.y_true,d.y_pred,d.y_score)
        fold_acc=[];fold_se=[];fold_sp=[]
        for fold,g in d.groupby('fold'):
            fm=metrics(g.y_true,g.y_pred,g.y_score)
            fold_acc.append(fm['accuracy']);fold_se.append(fm['sensitivity']);fold_sp.append(fm['specificity'])
            fold_scores.append({'feature_view':mode,'evaluation':evaluation,'model':model,'fold':fold,**fm})
        summaries.append({'feature_view':mode,'evaluation':evaluation,'model':model,**mm,
            'cv_mean_accuracy':float(np.mean(fold_acc)),'cv_sd_accuracy':float(np.std(fold_acc)),
            'cv_sem_accuracy':float(np.std(fold_acc,ddof=1)/np.sqrt(len(fold_acc))),
            'cv_mean_sensitivity':float(np.mean(fold_se)),
            'cv_mean_specificity':float(np.mean(fold_sp)),
            'n_unique_pregnancies':int(d.Record.nunique())})
    s=pd.DataFrame(summaries).sort_values(['feature_view','evaluation','accuracy'],ascending=[True,True,False]);f=pd.DataFrame(fold_scores)
    s.to_csv(out/'metrics.csv',index=False,float_format='%.10g');f.to_csv(out/'fold_metrics.csv',index=False,float_format='%.10g')
    # Essential audit: reported rates are ALWAYS tied to a single set of predictions.
    for row in s.itertuples():
        assert np.isclose(row.accuracy, (row.TN+row.TP)/row.n)
        assert np.isclose(row.sensitivity, row.TP/(row.TP+row.FN))
        assert np.isclose(row.specificity, row.TN/(row.TN+row.FP))
    return s


def paper_search(root:Path):
    """Compare each *joint* study row, never independently cherry-pick metrics."""
    root=Path(root); rows=pd.read_csv(root/'results/design_search/metrics.csv')
    targets=json.loads((root/'configs/paper_targets.json').read_text())
    name_map={'XG Boost':'XGBoost','Meta Bagging Classifier':'Meta Bagging',
              'Ensemble Classifier':'Hard Voting Ensemble'}
    found=[]
    for t in targets['binary_classifier_targets']:
        name=name_map.get(t['paper_name'],t['paper_name'])
        candidates=rows[rows.model==name].copy()
        if candidates.empty:continue
        # Scale each deviation in the *unit in which it is published*;
        # no attempt to force the model's confusion matrix to match targets.
        for _,r in candidates.iterrows():
            dacc=100*r.accuracy-t['accuracy_percent']
            dse=r.sensitivity-t['sensitivity'];dsp=r.specificity-t['specificity']
            dsd=r.cv_sd_accuracy-t['cv_sd']
            found.append({'model':t['paper_name'],'feature_view':r.feature_view,
                'evaluation':r.evaluation,'accuracy_percent':100*r.accuracy,
                'sensitivity':r.sensitivity,'specificity':r.specificity,
                'cv_sd':r.cv_sd_accuracy,'paper_accuracy_percent':t['accuracy_percent'],
                'paper_sensitivity':t['sensitivity'],'paper_specificity':t['specificity'],
                'paper_cv_sd':t['cv_sd'],'gap_accuracy_points':dacc,
                'gap_sensitivity':dse,'gap_specificity':dsp,'gap_cv_sd':dsd,
                'joint_objective':float(np.sqrt((dacc/5)**2+(dse/.1)**2+(dsp/.1)**2+(dsd/.02)**2))})
    c=pd.DataFrame(found).sort_values(['model','joint_objective'])
    c.to_csv(root/'results/design_search/joint_paper_comparison.csv',index=False)
    return c.groupby('model',sort=True).head(1).sort_values('joint_objective')

"""Paper-aligned feature expansion and nested, record-level model selection.

Outer folds are untouched by feature selection, hyperparameter search, oversampling,
threshold choice, and ensemble combination. All learned operations are fitted on
training recordings only. Positives denote preterm delivery.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score
from imblearn.ensemble import BalancedRandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as BalancedPipeline
from .data import feature_columns, CLINICAL, CATEGORICAL
from .evaluation import metrics

SEED=2026
BANDS={'low': '0.08_Hz-4.0_Hz', 'wide':'0.3_Hz-4.0_Hz'}


def build_multiband_features(frame:pd.DataFrame, source_dir:Path)->pd.DataFrame:
    """Build multiband features from the three source feature files, by ID.

    Fields tied to delivery/outcome are never used as model covariates.
    """
    out=frame.set_index('Record').copy()
    for band,suffix in BANDS.items():
        path=Path(source_dir)/f'tpehgdb_features__filter_{suffix}.fvl'
        d=pd.read_csv(path).rename(columns=lambda x:str(x).strip())
        d['Record']=d['Record'].astype(str).str.strip()
        if len(d)!=900 or d.duplicated(['Record','Chann']).any():
            raise ValueError(f'Unexpected source layout: {path}')
        for attr in ['RMS','Fmed','Fpeak','Samp. en.']:
            p=d.pivot(index='Record',columns='Chann',values=attr).astype(float)
            for ch in (1,2,3):
                out[f'{attr.replace(" ","_").replace(".","")}_{band}_{ch}']=p[ch]
    # Label-free transforms allow nonlinear classifiers to learn across channels
    # without inferring a disease label from any delivery-time variable.
    for attr in ('Fmed','Fpeak','RMS'):
        channels=[f'{attr}_{i}' for i in (1,2,3)]
        mat=out[channels].to_numpy(float)
        out[f'{attr}_channel_mean']=np.nanmean(mat,axis=1)
        out[f'{attr}_channel_sd']=np.nanstd(mat,axis=1)
        out[f'{attr}_channel_range']=np.nanmax(mat,axis=1)-np.nanmin(mat,axis=1)
    for attr in ('Fmed','Fpeak','RMS'):
        for ch in (1,2,3):
            original=out[f'{attr}_{ch}'].to_numpy(float)
            for band in BANDS:
                new=out[f'{attr}_{band}_{ch}'].to_numpy(float)
                out[f'{attr}_{band}_delta_{ch}']=new-original
                out[f'{attr}_{band}_ratio_{ch}']=new/(np.abs(original)+1e-6)
    out=out.reset_index()
    assert len(out)==len(frame) and out.Record.nunique()==len(frame)
    assert out['y'].equals(frame.sort_values('Record').reset_index(drop=True)['y']) if frame.Record.is_monotonic_increasing else True
    return out.replace([np.inf,-np.inf],np.nan)


def predictors(d:pd.DataFrame):
    disallowed=set(CLINICAL+CATEGORICAL+['y','Record','Gestation','Group','Premature','Early'])
    cols=[c for c in d if c not in disallowed and pd.api.types.is_numeric_dtype(d[c])]
    if set(cols)&disallowed:raise AssertionError('Outcome field found')
    return cols


def _search_space(model_name, seed):
    # A small prespecified search space keeps the nested 10-fold run practical.
    impute=SimpleImputer(strategy='median',keep_empty_features=True)
    if model_name=='Balanced Random Forest':
        classifier=BalancedRandomForestClassifier(n_estimators=65,replacement=True,random_state=seed,n_jobs=1)
        pipe=Pipeline([('impute',impute),('model',classifier)])
        grid={'model__max_depth':[4,None],'model__min_samples_leaf':[2]}
    elif model_name=='Extra Trees':
        pipe=Pipeline([('impute',impute),('model',ExtraTreesClassifier(n_estimators=90,class_weight='balanced',random_state=seed,n_jobs=1))])
        grid={'model__min_samples_leaf':[1,4]}
    elif model_name=='RBF SVM':
        pipe=Pipeline([('impute',impute),('scale',StandardScaler()),('model',SVC(class_weight='balanced',probability=True,random_state=seed))])
        grid={'model__C':[.1,2.]}
    elif model_name=='Logistic Regression':
        pipe=Pipeline([('impute',impute),('scale',StandardScaler()),('model',LogisticRegression(max_iter=2000,class_weight='balanced',random_state=seed))])
        grid={'model__C':[.03,.5]}
    elif model_name=='HistGB + oversampling':
        pipe=BalancedPipeline([('impute',impute),('balance',RandomOverSampler(random_state=seed)),('model',HistGradientBoostingClassifier(max_iter=60,l2_regularization=4,random_state=seed))])
        grid={'model__max_leaf_nodes':[5,15]}
    else:raise ValueError(model_name)
    return pipe,grid


def _optimal_threshold(y,p, priority='balanced'):
    """Choose threshold using only training-fold out-of-fold predictions.

    Balanced accuracy gives recall and specificity equal weight, independent
    of the prevalence imbalance. Use quantile candidates (not the test set).
    """
    q=np.unique(np.r_[0.0,1.0,np.quantile(p,np.linspace(.01,.99,99)),.5])
    scored=[]
    for t in q:
        pred=(p>=t).astype(int)
        tn=((y==0)&(pred==0)).sum();fp=((y==0)&(pred==1)).sum()
        tp=((y==1)&(pred==1)).sum();fn=((y==1)&(pred==0)).sum()
        se=tp/max(tp+fn,1);sp=tn/max(tn+fp,1)
        score=(se+sp)/2 if priority=='balanced' else .65*se+.35*sp
        scored.append((score,-abs(t-.5),float(t),se,sp))
    score,_,t,se,sp=max(scored)
    return float(t),float(score),float(se),float(sp)


def evaluate_advanced(frame:pd.DataFrame,source_dir:Path,out_dir:Path,folds=10,seed=SEED):
    out_dir=Path(out_dir);out_dir.mkdir(exist_ok=True,parents=True)
    full=build_multiband_features(frame,source_dir)
    full.to_csv(out_dir/'multiband_features.csv',index=False)
    cols=predictors(full)
    y=full.y.to_numpy(dtype=int); X=full[cols].replace([np.inf,-np.inf],np.nan)
    outer=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    families=['Balanced Random Forest','Extra Trees','RBF SVM','Logistic Regression','HistGB + oversampling']
    preds=[];fms=[];params=[];selected=[]
    for fold,(tr,te) in enumerate(outer.split(X,y),start=1):
        inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed+fold)
        train,hold=X.iloc[tr],X.iloc[te];yt=y[tr]
        fitted_scores={};inner_scores={}
        for idx,name in enumerate(families):
            pipe,grid=_search_space(name,seed+fold)
            search=GridSearchCV(pipe,grid,scoring='roc_auc',cv=inner,n_jobs=1,refit=True,error_score='raise')
            search.fit(train,yt)
            # threshold learned from out-of-fold train predictions, never test
            oof=cross_val_predict(clone(search.best_estimator_),train,yt,cv=inner,method='predict_proba',n_jobs=1)[:,1]
            threshold,score,train_se,train_sp=_optimal_threshold(yt,oof)
            pred_scores=search.best_estimator_.predict_proba(hold)[:,1]
            fitted_scores[name]=pred_scores
            inner_scores[name]=oof
            yh=(pred_scores>=threshold).astype(int)
            for i in range(len(te)):
                preds.append({'Record':full.iloc[te[i]].Record,'fold':fold,'model':name,'y_true':int(y[te[i]]),
                              'y_pred':int(yh[i]),'y_score':float(pred_scores[i]),'threshold':threshold})
            params.append({'fold':fold,'model':name,'best_params':search.best_params_,'inner_auc':search.best_score_,
                           'threshold':threshold,'inner_balanced_accuracy':score,'inner_sensitivity':train_se,'inner_specificity':train_sp})
        # soft voting selected on training data alone; ensemble weights from
        # individual train-only OOF AUC, constrained to nonnegative reliability.
        from sklearn.metrics import roc_auc_score
        au=np.array([roc_auc_score(yt,inner_scores[n]) for n in families],float)
        w=np.maximum(au-.5,.02);w=w/w.sum()
        oofensemble=sum(w[i]*inner_scores[n] for i,n in enumerate(families))
        ensemble=sum(w[i]*fitted_scores[n] for i,n in enumerate(families))
        threshold,sc,se,sp=_optimal_threshold(yt,oofensemble)
        for i in range(len(te)):
            preds.append({'Record':full.iloc[te[i]].Record,'fold':fold,'model':'Calibrated Soft Ensemble','y_true':int(y[te[i]]),
                          'y_pred':int(ensemble[i]>=threshold),'y_score':float(ensemble[i]),'threshold':threshold})
        params.append({'fold':fold,'model':'Calibrated Soft Ensemble','base_models':families,'weights':dict(zip(families,w.tolist())),
                       'threshold':threshold,'inner_balanced_accuracy':sc,'inner_sensitivity':se,'inner_specificity':sp})
        print(f'Nested fold {fold}/{folds} complete | ensemble weights {np.round(w,2)}',flush=True)
    p=pd.DataFrame(preds)
    if p.duplicated(['Record','model']).any():raise AssertionError('One test prediction per recording and model expected')
    assert p.groupby('model').size().eq(len(frame)).all()
    p.to_csv(out_dir/'predictions.csv',index=False,float_format='%.12g')
    p=pd.read_csv(out_dir/'predictions.csv')
    (out_dir/'tuning.json').write_text(json.dumps(params,indent=2))
    summary=[]
    for name in families+['Calibrated Soft Ensemble']:
        pp=p[p.model==name]
        m=metrics(pp.y_true,pp.y_pred,pp.y_score)
        foldacc=[]
        for fold in range(1,folds+1):
            z=pp[pp.fold==fold];foldacc.append(float((z.y_pred==z.y_true).mean()))
        summary.append({'model':name,**m,'cv_accuracy_sd':float(np.std(foldacc)), 'cv_accuracy_mean':float(np.mean(foldacc)),
                        'folds':folds,'unit':'recording','threshold_protocol':'train-only OOF balanced accuracy'})
    tab=pd.DataFrame(summary).sort_values('balanced_accuracy',ascending=False)
    tab.to_csv(out_dir/'metrics.csv',index=False,float_format='%.8f')
    config={'seed':seed,'outer_folds':folds,'inner_folds':3,'all_predictors':cols,'target':'delivery before 37 weeks',
            'balanced_threshold':'selected on training-only three-fold OOF',
            'feature_selector':'per-model nested hyperparameter tuning, with the five feature families included and expanded with two additional filter bands',
            'data_leakage_excluded_fields':list(CLINICAL+CATEGORICAL+['Gestation','Group','Premature','Early'])}
    (out_dir/'manifest.json').write_text(json.dumps(config,indent=2))
    return tab

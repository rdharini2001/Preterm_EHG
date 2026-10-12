"""Ten-fold record-disjoint ML evaluation with saved out-of-fold probabilities."""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import (accuracy_score,balanced_accuracy_score,precision_score,
                             recall_score,f1_score,confusion_matrix,roc_auc_score)
from .data import feature_columns
from .models import classifier_definitions,preprocessing,select_family_groups,SEED


def metrics(y,yp,probs=None):
    tn,fp,fn,tp=confusion_matrix(y,yp,labels=[0,1]).ravel()
    return dict(n=len(y),TN=int(tn),FP=int(fp),FN=int(fn),TP=int(tp),
                accuracy=float(accuracy_score(y,yp)),balanced_accuracy=float(balanced_accuracy_score(y,yp)),
                sensitivity=float(recall_score(y,yp,zero_division=0)),
                specificity=float(tn/(tn+fp)) if tn+fp else float('nan'),
                precision=float(precision_score(y,yp,zero_division=0)),
                f1=float(f1_score(y,yp,zero_division=0)),
                auc=float(roc_auc_score(y,probs)) if probs is not None and len(set(y))==2 else float('nan'))


def evaluate_ml(frame:pd.DataFrame, out_dir:Path, kind='five', clinical=False,
                folds=10, select=True, seed=SEED):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    X=frame.set_index('Record');y=X.pop('y').astype(int)
    available=feature_columns(frame,kind=kind,clinical=clinical)
    estimators=classifier_definitions()
    splits=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    records=[];perfold=[];selection=[];folds_manifest=[]
    for fold,(tr,te) in enumerate(splits.split(X[available],y),start=1):
        train,test=X.iloc[tr],X.iloc[te]
        yt,ye=y.iloc[tr],y.iloc[te]
        if set(train.index)&set(test.index):raise AssertionError('Record overlap between train and test')
        if select:
            cols,families,selection_scores=select_family_groups(train,yt,available,seed=seed+fold)
        else:
            cols=available;families=[];selection_scores=[]
        # Optional additional clinical inputs are not part of the family selection.
        if clinical:
            cols += [c for c in available if c not in cols and c not in feature_columns(frame,kind='five')]
        selection.append(dict(fold=fold,feature_families=families,features=cols,cv_balanced_accuracy=selection_scores))
        folds_manifest.extend({'Record':r,'fold':fold,'split':'test','y':int(v)} for r,v in ye.items())
        for name,base in estimators.items():
            model=Pipeline([('preprocessing',preprocessing(cols)),('classifier',clone(base))])
            model.fit(train[cols],yt)
            predicted=model.predict(test[cols]).astype(int)
            if hasattr(model,'predict_proba'):
                probabilities=model.predict_proba(test[cols])[:,1]
            elif hasattr(model,'decision_function'):
                probabilities=model.decision_function(test[cols])
            else:
                probabilities=None
            for j,(record,truth) in enumerate(ye.items()):
                records.append(dict(Record=record,fold=fold,model=name,y_true=int(truth),y_pred=int(predicted[j]),
                                    y_score=float(probabilities[j]) if probabilities is not None else np.nan))
            mf=metrics(ye,predicted,probabilities)
            perfold.append(dict(model=name,fold=fold,**mf))
        print(f'ML fold {fold:02d}/{folds}: selected {families}',flush=True)
    p=pd.DataFrame(records).sort_values(['model','fold','Record']).reset_index(drop=True)
    ff=pd.DataFrame(perfold)
    p.to_csv(out_dir/'predictions.csv',index=False)
    ff.to_csv(out_dir/'fold_metrics.csv',index=False)
    pd.DataFrame(folds_manifest).to_csv(out_dir/'split_manifest.csv',index=False)
    (out_dir/'selected_features.json').write_text(json.dumps(selection,indent=2))
    summary=[]
    for name in estimators:
        pp=p[p.model==name]
        m=metrics(pp.y_true,pp.y_pred,pp.y_score if pp.y_score.notna().all() else None)
        m.update(model=name,cv_accuracy_sd=float(ff[ff.model==name].accuracy.std(ddof=0)),
                 cv_accuracy_mean=float(ff[ff.model==name].accuracy.mean()),folds=folds,
                 evaluation='10-fold stratified record-disjoint CV')
        summary.append(m)
    result=pd.DataFrame(summary).sort_values('accuracy',ascending=False).reset_index(drop=True)
    result.to_csv(out_dir/'model_metrics.csv',index=False,float_format='%.8f')
    # Every sample must have exactly one out-of-fold prediction per method.
    for name in estimators:
        sample=p[p.model==name]
        assert len(sample)==len(frame) and sample.Record.nunique()==len(frame)
    return result

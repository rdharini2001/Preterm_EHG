"""Paper-named classifiers fitted on actual signal-window feature tables.

Two explicitly different prediction tasks are evaluated: within-record window CV
(which has overlapping training/test signal segments) and entire-record holdout.
Models, probabilities, true labels, and all held-out identifiers are saved.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold,StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import AdaBoostClassifier


def get_models(seed:int):
    return {
      'kNN (k=11)':KNeighborsClassifier(n_neighbors=11,weights='uniform',n_jobs=1),
      'kNN (k=31)':KNeighborsClassifier(n_neighbors=31,weights='distance',n_jobs=1),
      'Logistic Regression':LogisticRegression(max_iter=1000,class_weight='balanced',random_state=seed),
      'Naive Bayes':GaussianNB(var_smoothing=1e-8),
      'Support Vector Machines':LinearSVC(C=0.3,class_weight='balanced',max_iter=2000,random_state=seed,dual=False),
      'Adaptive Boost':AdaBoostClassifier(n_estimators=45,learning_rate=.7,random_state=seed),
    }


def run(root:Path,seed:int=2026,n_folds:int=10):
    root=Path(root)
    f=pd.read_csv(root/'data/processed/window_features.csv.gz')
    cols=[x for x in f if x not in ['Record','window','start_sample','y']]
    x=f[cols].to_numpy(float);y=f.y.to_numpy(int);group=f.Record.to_numpy()
    outdir=root/'results/window_classifiers';outdir.mkdir(parents=True,exist_ok=True)
    records=[];audit=[]
    for evaluation,splitter in [('window_stratified',StratifiedKFold(n_splits=n_folds,shuffle=True,random_state=seed)),
                                ('recording_disjoint',StratifiedGroupKFold(n_splits=n_folds,shuffle=True,random_state=seed))]:
        splits=(splitter.split(x,y) if evaluation=='window_stratified' else splitter.split(x,y,group))
        for k,(tr,te) in enumerate(splits,1):
            shared=len(set(group[tr])&set(group[te]));audit.append(dict(evaluation=evaluation,fold=k,
                              n_train=len(tr),n_test=len(te),shared_recordings=shared))
            if evaluation=='recording_disjoint' and shared:raise ValueError('Leakage in recording holdout')
            for model_name,est in get_models(seed).items():
                pipe=make_pipeline(SimpleImputer(strategy='median'),StandardScaler(),clone(est))
                pipe.fit(x[tr],y[tr])
                if hasattr(pipe,'predict_proba'): score=pipe.predict_proba(x[te])[:,1]
                else: score=expit(pipe.decision_function(x[te]))
                for i,prob in zip(te,score):
                    records.append((evaluation,k,group[i],int(f.window.iloc[i]),model_name,int(y[i]),float(prob)))
        print(f'Completed {evaluation}',flush=True)
    colsout=['evaluation','fold','Record','window','model','y_true','y_score']
    pd.DataFrame.from_records(records,columns=colsout).to_csv(outdir/'predictions.csv',index=False,float_format='%.12g')
    pd.DataFrame(audit).to_csv(outdir/'split_audit.csv',index=False)
    (outdir/'config.json').write_text(json.dumps(dict(seed=seed,folds=n_folds,
       feature_columns=cols,objective='term/preterm from 750-sample EHG windows',
       tasks=['window_stratified','recording_disjoint'],
       caveat='Window-stratified folds contain correlated and overlapping windows from a single recording.'),indent=2)+'\n')
    return outdir

"""Classifiers and leakage-free selection from train-set observations only."""
from __future__ import annotations
import numpy as np
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import (RandomForestClassifier,AdaBoostClassifier,
                              BaggingClassifier,GradientBoostingClassifier,VotingClassifier)
from xgboost import XGBClassifier
from .data import CATEGORICAL, FEATURE_FAMILIES

SEED=2026


def preprocessing(columns):
    numeric=[x for x in columns if x not in CATEGORICAL]
    cats=[x for x in columns if x in CATEGORICAL]
    transformers=[('numeric',Pipeline([('impute',SimpleImputer(strategy='median',keep_empty_features=True)),
                                       ('scale', StandardScaler())]),numeric)]
    if cats:
        transformers.append(('categorical',Pipeline([('impute',SimpleImputer(strategy='most_frequent')),
                       ('encode',OneHotEncoder(handle_unknown='ignore',sparse_output=False))]),cats))
    return ColumnTransformer(transformers,sparse_threshold=0)


def classifier_definitions():
    base={
       'kNN':KNeighborsClassifier(n_neighbors=7,weights='distance'),
       'SVM':SVC(C=1.0,gamma='scale',probability=True,class_weight='balanced',random_state=SEED),
       'AdaBoost':AdaBoostClassifier(n_estimators=60,learning_rate=.6,random_state=SEED),
       'Logistic Regression':LogisticRegression(max_iter=1500,class_weight='balanced',random_state=SEED),
       'Naive Bayes':GaussianNB(),
       'Decision Tree':DecisionTreeClassifier(max_depth=7,min_samples_leaf=2,class_weight='balanced',random_state=SEED),
       'XGBoost':XGBClassifier(n_estimators=100,max_depth=3,learning_rate=.06,
                               subsample=.9,colsample_bytree=.9,reg_lambda=2,n_jobs=1,
                               eval_metric='logloss',random_state=SEED),
       'Meta Bagging':BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=8,min_samples_leaf=2,
                                       class_weight='balanced',random_state=SEED),n_estimators=75,random_state=SEED,n_jobs=1),
       'Gradient Boost':GradientBoostingClassifier(n_estimators=100,learning_rate=.05,max_depth=2,random_state=SEED),
       'Random Forest':RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=2,
                                             class_weight='balanced_subsample',n_jobs=1,random_state=SEED),
    }
    names=['Decision Tree','Random Forest','XGBoost','Meta Bagging','Gradient Boost']
    base['Hard Voting Ensemble']=VotingClassifier([(x.replace(' ','_'),clone(base[x])) for x in names],voting='hard')
    return base


def select_family_groups(X,y,available,seed=SEED,max_groups=5):
    """Sequential forward selection, ranked by inner-fold balanced accuracy.

    A family contains the same statistic on three EHG channels. The inner CV
    uses *only outer training records*. For small N each addition is measured
    using the same inner split; improvement <0.002 stops selection.
    """
    groups={family:[f'{family}_{j}' for j in (1,2,3)] for family in FEATURE_FAMILIES}
    groups={k:v for k,v in groups.items() if all(f in available for f in v)}
    if not groups:raise ValueError('None of the five paper feature families are available')
    inner=StratifiedKFold(n_splits=3,shuffle=True,random_state=seed)
    selected=[];scores=[];best=-1.
    for _ in range(min(max_groups,len(groups))):
        candidates=[]
        for group in groups:
            if group in selected:continue
            names=[col for g in selected+[group] for col in groups[g]]
            estimator=Pipeline([('pre',preprocessing(names)),('model',LogisticRegression(max_iter=1000,class_weight='balanced',random_state=seed))])
            val=cross_val_score(estimator,X[names],y,cv=inner,scoring='balanced_accuracy',n_jobs=1).mean()
            candidates.append((float(val),group))
        score,name=max(candidates,key=lambda p:(p[0],-list(groups).index(p[1])))
        if selected and score<=best+.002:break
        selected.append(name);scores.append(score);best=score
    return [c for g in selected for c in groups[g]], selected, scores

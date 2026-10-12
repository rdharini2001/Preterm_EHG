"""Fit extra paper-family variants on EHG windows for protocol investigation.

Window-stratified cross-validation permits training/test windows from the same
recording; these predictions are not recording-independent performance estimates.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


def run(root:Path):
    d=pd.read_csv(root/'data/processed/window_features.csv.gz')
    X=d.drop(columns=['Record','window','start_sample','y']).to_numpy(float)
    y=d.y.to_numpy(int)
    models={
       'kNN (k=71)':make_pipeline(StandardScaler(),KNeighborsClassifier(n_neighbors=71,weights='distance',n_jobs=1)),
       'Decision Tree (depth=8)':DecisionTreeClassifier(max_depth=8,min_samples_leaf=2,random_state=2026),
       'Decision Tree (depth=12)':DecisionTreeClassifier(max_depth=12,min_samples_leaf=3,random_state=2026),
       'Decision Tree (depth=16)':DecisionTreeClassifier(max_depth=16,min_samples_leaf=1,random_state=2026),
    }
    rows=[]
    for name,model in models.items():
        for fold,(train,test) in enumerate(StratifiedKFold(n_splits=10,shuffle=True,random_state=2026).split(X,y),1):
            model.fit(X[train],y[train])
            score=model.predict_proba(X[test])[:,1]
            for i,prob in zip(test,score):
                rows.append((name,fold,str(d.Record.iloc[i]),int(d.window.iloc[i]),int(y[i]),float(prob)))
        print('Completed',name,flush=True)
    target=root/'results/window_architecture_sweep';target.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows,columns=['model','fold','Record','window','y_true','y_score']).assign(
        evaluation='window_stratified_10fold').to_csv(target/'predictions.csv',index=False)

if __name__=='__main__':run(Path(__file__).resolve().parents[1])

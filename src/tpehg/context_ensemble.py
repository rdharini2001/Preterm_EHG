"""EHG multi-window signal-context classifiers.

Record context contains signal summary statistics extracted without outcome
labels. Two cross-validation populations are evaluated without mixing them:
* within_record_windows: tests new windows from observed pregnancy recordings;
* unseen_recordings: tests windows from recordings absent during training.

The paper's five-tree hard-voting classifier is implemented, using a single
scikit-learn-style decision per base model. Model choice is preconfigured.
"""
from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier,HistGradientBoostingClassifier
from sklearn.metrics import confusion_matrix,accuracy_score
from xgboost import XGBClassifier
from .evaluation import metrics

META = {'Record', 'window', 'start_sample', 'y'}


def prepare_context(windows:pd.DataFrame,record_features:pd.DataFrame):
    """Append label-free EHG recording descriptors to 750/125 windows.

    The source's record-level features are calculated from the *full signal*.
    They are legitimate observed-record inputs, but they must not be confused
    with newly unseen-pregnancy prediction when evaluating overlapping windows.
    """
    protected = {'Record','y','Rectime','Age','Parity','Abortions','Weight','Premature','Group','Gestation','Early'}
    numeric = [c for c in record_features.columns
               if c not in protected and pd.api.types.is_numeric_dtype(record_features[c])]
    if any('Group' in c or 'Gestation' in c or 'Premature' in c for c in numeric):
        raise ValueError('Outcome-derived column in signal context')
    context = record_features[['Record']+numeric].rename(columns={c:'context_'+c for c in numeric})
    out = windows.merge(context,on='Record',how='left',validate='many_to_one')
    if len(out)!=len(windows):raise AssertionError('Record context cardinality changed')
    y_map=record_features.set_index('Record').y
    if not (out.y.values==out.Record.map(y_map).values).all():raise AssertionError('Label misalignment')
    return out


def trees(seed:int=2026):
    # All five classifier families explicitly named in the paper.
    return {
      'Decision Tree':DecisionTreeClassifier(max_depth=7,min_samples_leaf=2,random_state=seed),
      'Random Forest':RandomForestClassifier(n_estimators=20,max_depth=8,n_jobs=1,random_state=seed),
      'XGBoost':XGBClassifier(n_estimators=30,max_depth=4,learning_rate=.10,subsample=.9,
                             colsample_bytree=.9,n_jobs=1,random_state=seed,eval_metric='logloss'),
      'Meta Bagging':BaggingClassifier(estimator=DecisionTreeClassifier(max_depth=7,min_samples_leaf=2),
                                     n_estimators=12,n_jobs=1,random_state=seed),
      'Gradient Boost':HistGradientBoostingClassifier(max_iter=35,max_leaf_nodes=8,learning_rate=.09,random_state=seed),
    }


def run_context_ensemble(windows:pd.DataFrame,records:pd.DataFrame,out_dir:Path,folds:int=10,seed:int=2026,
                         modes=('within_record_windows','unseen_recordings')):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    frame=prepare_context(windows,records)
    columns=[c for c in frame if c not in META and (c.startswith('context_') or c in ['Fmed_1','Fmed_2','Fmed_3','Fpeak_1','Fpeak_2','Fpeak_3','RMS_1','RMS_2','RMS_3','zero_crossing_1','zero_crossing_2','zero_crossing_3','variance_1','variance_2','variance_3'])]
    x=frame[columns].to_numpy(float);y=frame.y.to_numpy(int)
    groups=frame.Record.to_numpy();window=frame.window.to_numpy(int)
    if len(frame)!=7200 or len(set(groups))!=300:raise ValueError('Expected 24 windows from each of 300 records')
    all_p=[]; fold_audit=[]
    for protocol in modes:
        if protocol=='within_record_windows':splits=StratifiedKFold(folds,shuffle=True,random_state=seed).split(x,y)
        elif protocol=='unseen_recordings':splits=StratifiedGroupKFold(folds,shuffle=True,random_state=seed).split(x,y,groups)
        else:raise ValueError(protocol)
        for fold,(tr,te) in enumerate(splits,1):
            train_records=set(groups[tr]);test_records=set(groups[te]);shared=len(train_records & test_records)
            if protocol=='unseen_recordings' and shared:raise AssertionError('Recording overlap')
            fold_audit.append({'protocol':protocol,'fold':fold,'train_windows':len(tr),'test_windows':len(te),
                               'train_recordings':len(train_records),'test_recordings':len(test_records),
                               'recordings_in_both':shared})
            # Preprocessing fits training data only.
            imputer=SimpleImputer(strategy='median');xx=imputer.fit_transform(x[tr]);xt=imputer.transform(x[te]);votes=[]
            for name,model in trees(seed).items():
                model.fit(xx,y[tr]);p=model.predict(xt).astype(int)
                scores=model.predict_proba(xt)[:,1]
                votes.append(p)
                all_p.extend({'protocol':protocol,'fold':fold,'Record':groups[idx],
                    'window':int(window[idx]),'model':name,'y_true':int(y[idx]),
                    'y_pred':int(pred),'y_score':float(sc)} for idx,pred,sc in zip(te,p,scores))
            # Five odd-numbered classifiers: no plurality ties are possible.
            voting=(np.asarray(votes).sum(axis=0)>=3).astype(int)
            all_p.extend({'protocol':protocol,'fold':fold,'Record':groups[idx],
                'window':int(window[idx]),'model':'Hard Voting Ensemble',
                'y_true':int(y[idx]),'y_pred':int(pred),'y_score':float(sum(v[idx2] for v in votes)/5.)}
                for idx2,(idx,pred) in enumerate(zip(te,voting)))
            print(f'Context ensemble {protocol} fold {fold}/{folds}',flush=True)
    p=pd.DataFrame(all_p).sort_values(['protocol','model','fold','Record','window'])
    p.to_csv(out_dir/'predictions.csv',index=False,float_format='%.10g')
    p=pd.read_csv(out_dir/'predictions.csv')
    summary=[]; by_fold=[]
    for (protocol,model),part in p.groupby(['protocol','model'],sort=True):
        if len(part)!=7200:raise AssertionError(f'{protocol} {model} incomplete: {len(part)}')
        if part[['Record','window']].duplicated().any():raise AssertionError('Duplicate prediction')
        m=metrics(part.y_true,part.y_pred,part.y_score);m.update(protocol=protocol,model=model,
            prediction_unit='window',recordings=300,folds=folds)
        grouped=[]
        for fi,pf in part.groupby('fold'):
            fm=metrics(pf.y_true,pf.y_pred,pf.y_score)
            fm.update(protocol=protocol,model=model,fold=fi);by_fold.append(fm);grouped.append(fm['accuracy'])
        m.update(cv_mean_accuracy=float(np.mean(grouped)),cv_sd_accuracy=float(np.std(grouped,ddof=0)))
        summary.append(m)
    scores=pd.DataFrame(summary).sort_values(['protocol','accuracy'],ascending=[True,False])
    scores.to_csv(out_dir/'metrics.csv',index=False,float_format='%.8f')
    pd.DataFrame(by_fold).to_csv(out_dir/'fold_metrics.csv',index=False,float_format='%.8f')
    pd.DataFrame(fold_audit).to_csv(out_dir/'split_audit.csv',index=False)
    (out_dir/'config.json').write_text(json.dumps({'seed':seed,'folds':folds,'window_size':750,'shift':125,
        'samples_per_record':24,'features':columns,'models':{k:str(v) for k,v in trees(seed).items()},
        'protocols':{
        'within_record_windows':'Stratified window folds; recording-level signal contexts reused across windows; observed-record task',
        'unseen_recordings':'Stratified group folds; contexts for held-out recordings never seen in training; new-record task'}},indent=2))
    return scores

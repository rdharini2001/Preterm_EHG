"""750-sample / 125-shift EHG window feature extraction and evaluations.

Two separate analyses are provided: window-stratified folds (predict a window
from partially observed recordings) and recording-stratified folds (predict
windows from new recordings). These estimate different prediction tasks.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy.signal import welch
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.ensemble import RandomForestClassifier,ExtraTreesClassifier,HistGradientBoostingClassifier,VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from imblearn.ensemble import BalancedRandomForestClassifier
from sklearn.metrics import accuracy_score
from .signals import load_filtered_record
from .evaluation import metrics


def extract_windows(raw_dir:Path, label_table:pd.DataFrame, window=750, shift=125, blocks=4, per_block=6):
    labels=label_table.set_index('Record')['y']
    rows=[]
    for i,record in enumerate(labels.index):
        array,fs=load_filtered_record(Path(raw_dir)/f'{record}.dat')
        max_start=array.shape[1]-window-((per_block-1)*shift)
        if max_start<0:raise ValueError('Short recording')
        anchors=np.linspace(0,max_start,blocks,dtype=int)
        for block,anchor in enumerate(anchors):
            for k in range(per_block):
                start=int(anchor+k*shift)
                windows=array[:,start:start+window].astype('float64')
                # Welch estimates are made per channel; cached waveforms are prefiltered.
                f,p=welch(windows,fs=fs,nperseg=750,axis=1)
                inds=(f>=0.05)&(f<=3)
                f=f[inds];p=p[:,inds]
                density=p/np.maximum(p.sum(axis=1,keepdims=True),1e-12)
                cum=np.cumsum(density,axis=1)
                r={'Record':record,'window':block*per_block+k,'start_sample':start,'y':int(labels[record])}
                for ch in range(3):
                    x=windows[ch];v=x.var();power=p[ch];d=density[ch]
                    for name,value in {
                        'Fmed':f[min(np.searchsorted(cum[ch],.5), len(f)-1)],'Fpeak':f[np.argmax(power)],
                        'RMS':np.sqrt(np.mean(x*x)),'zero_crossing':np.sum(x[1:]*x[:-1]<0)/(window/fs),
                        'variance':v,'spectral_entropy':-np.sum(d*np.log(np.maximum(d,1e-12))),
                        'spectral_centroid':np.sum(f*d),'hjorth_mobility':np.sqrt(np.diff(x).var()/(v+1e-12)),
                        'bandpower_low':power[(f>=.05)&(f<.3)].sum()/(power.sum()+1e-12),
                        'bandpower_mid':power[(f>=.3)&(f<1)].sum()/(power.sum()+1e-12),
                        'bandpower_high':power[(f>=1)&(f<=3)].sum()/(power.sum()+1e-12),
                    }.items():r[f'{name}_{ch+1}']=float(value)
                rows.append(r)
        if (i+1)%75==0:print(f'Extracted EHG windows for {i+1}/{len(labels)} recordings',flush=True)
    out=pd.DataFrame(rows)
    assert out.Record.nunique()==300 and len(out)==len(label_table)*blocks*per_block
    return out


def evaluate_windows(frame:pd.DataFrame,out_dir:Path,folds=5,seed=2026):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    meta=['Record','window','start_sample','y']
    cols=[x for x in frame if x not in meta]
    x=frame[cols].to_numpy(float);y=frame.y.to_numpy(int);groups=frame.Record.to_numpy()
    models={
        'Random Forest':RandomForestClassifier(n_estimators=85,n_jobs=1,max_features='sqrt',random_state=seed),
        'Extra Trees':ExtraTreesClassifier(n_estimators=85,n_jobs=1,max_features='sqrt',random_state=seed),
        'Hist Gradient Boost':HistGradientBoostingClassifier(max_iter=75,max_leaf_nodes=25,random_state=seed),
        'Balanced Random Forest':BalancedRandomForestClassifier(n_estimators=75,min_samples_leaf=2,random_state=seed,n_jobs=1),
        '1-NN Window Matching':KNeighborsClassifier(n_neighbors=1,weights='distance',n_jobs=1),
    }
    all_preds=[]; all_summary=[]
    for protocol,cv in [('window_stratified',StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)),
                        ('record_disjoint',StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed))]:
        split_iter=cv.split(x,y) if protocol=='window_stratified' else cv.split(x,y,groups)
        for fold,(tr,te) in enumerate(split_iter,1):
            if protocol=='record_disjoint' and set(groups[tr])&set(groups[te]):raise AssertionError('Record overlap')
            for name,est in models.items():
                steps=[('impute',SimpleImputer(strategy='median'))]
                if name=='1-NN Window Matching':steps.append(('scale',StandardScaler()))
                steps.append(('estimator',est))
                clf=Pipeline(steps)
                clf.fit(x[tr],y[tr]);score=clf.predict_proba(x[te])[:,1];pred=(score>=.5).astype(int)
                for ix,s,pr in zip(te,score,pred):
                    all_preds.append({'evaluation':protocol,'fold':fold,'Record':groups[ix],
                                      'window':int(frame.iloc[ix].window),'model':name,'y_true':int(y[ix]),
                                      'y_score':float(s),'y_pred':int(pr)})
            print(f'{protocol}: fold {fold}/{folds}',flush=True)
    p=pd.DataFrame(all_preds);p.to_csv(out_dir/'predictions.csv',index=False,float_format='%.12g');p=pd.read_csv(out_dir/'predictions.csv')
    for (protocol,name),part in p.groupby(['evaluation','model'],sort=False):
        m=metrics(part.y_true,part.y_pred,part.y_score)
        per_fold=part.groupby('fold').apply(lambda d:accuracy_score(d.y_true,d.y_pred),include_groups=False)
        m.update(evaluation=protocol,model=name,folds=folds,cv_accuracy_sd=float(per_fold.std(ddof=0)),
                 prediction_unit='EHG window',n_independent_recordings=frame.Record.nunique())
        all_summary.append(m)
    out=pd.DataFrame(all_summary).sort_values(['evaluation','balanced_accuracy'],ascending=[True,False])
    out.to_csv(out_dir/'metrics.csv',index=False,float_format='%.8f')
    (out_dir/'config.json').write_text(json.dumps({'seed':seed,'window_samples':750,'shift_samples':125,'blocks':4,
         'windows_per_block':6,'folds':folds,'feature_columns':cols,
         'window_stratified':'Random windows assigned to folds; recordings occur in both train and test',
         'record_disjoint':'Whole recordings held out from train/validation splits'},indent=2))
    return out


def overlap_audit(frame:pd.DataFrame,out_dir:Path,folds=5,seed=2026,window=750):
    """Count test windows sharing literal time samples with train windows of the same recording."""
    x=frame.drop(columns=['Record','window','start_sample','y']).to_numpy(float)
    y=frame.y.to_numpy(int);group=frame.Record.to_numpy();starts=frame.start_sample.to_numpy(int)
    out=[]
    for evaluation,cv in [('window_stratified',StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)),
                          ('record_disjoint',StratifiedGroupKFold(n_splits=folds,shuffle=True,random_state=seed))]:
        ix=cv.split(x,y) if evaluation=='window_stratified' else cv.split(x,y,group)
        for fold,(tr,te) in enumerate(ix,1):
            train_by_record={r:starts[tr][group[tr]==r] for r in np.unique(group[tr])}
            overlap=[]
            for i in te:
                existing=train_by_record.get(group[i],[])
                overlap.append(bool(len(existing) and np.any(np.abs(existing-starts[i])<window)))
            out.append({'evaluation':evaluation,'fold':fold,'n_windows':len(te),
                        'n_overlapping_with_train':int(sum(overlap)),
                        'pct_overlapping_with_train':float(np.mean(overlap))*100,
                        'n_recordings_test':len(set(group[te])),
                        'n_recordings_shared_train_test':len(set(group[te])&set(group[tr]))})
    report=pd.DataFrame(out)
    report.to_csv(Path(out_dir)/'overlap_audit.csv',index=False,float_format='%.8f')
    return report

"""Seed-stability comparison for a fixed random-forest architecture."""
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import RandomOverSampler
from .context_ensemble import trees
from .data import build_dataset
from .design_search import get_X,get_splits
from .evaluation import metrics


def run(root:Path,seeds=(7,42,2026),folds=10):
    root=Path(root); out=root/'results/stability';out.mkdir(parents=True,exist_ok=True)
    frame=build_dataset(root/'data/source',root/'data/raw/tpehgdb',root/'data/processed/signal_features.csv')
    x,_=get_X(frame,root/'data/source','three_frequency_bands');y0=frame.y.to_numpy(int);ids=frame.Record.to_numpy()
    ps=[];rows=[]
    for seed in seeds:
        for evaluation in ('record_cv','train_fold_oversampling','pre_cv_oversampling_diagnostic'):
            for fold,tr,te,X,y,origin in get_splits(x,y0,ids,evaluation,folds,seed):
                imp=SimpleImputer(strategy='median',keep_empty_features=True)
                xx=imp.fit_transform(X[tr]);xt=imp.transform(X[te]);yt=y[tr]
                if evaluation=='train_fold_oversampling':
                    xx,yt=RandomOverSampler(random_state=seed+fold).fit_resample(xx,yt)
                m=trees(seed+fold)['Random Forest'];m.fit(xx,yt)
                pred=m.predict(xt).astype(int);prob=m.predict_proba(xt)[:,1]
                ps.extend({'seed':seed,'evaluation':evaluation,'fold':fold,'Record':str(origin[i]),
                           'y_true':int(y[i]),'y_pred':int(p),'y_score':float(s)} for i,p,s in zip(te,pred,prob))
            print(f'Completed stability seed={seed} {evaluation}',flush=True)
    p=pd.DataFrame(ps);p.to_csv(out/'predictions.csv',index=False,float_format='%.10g')
    for (seed,evaluation),g in p.groupby(['seed','evaluation']):
        met=metrics(g.y_true,g.y_pred,g.y_score)
        accs=[(f.y_pred==f.y_true).mean() for _,f in g.groupby('fold')]
        rows.append({'seed':seed,'evaluation':evaluation,**met,'cv_sd_accuracy':float(np.std(accs)),
                     'unique_pregnancies':g.Record.nunique()})
    s=pd.DataFrame(rows).sort_values(['evaluation','seed'])
    s.to_csv(out/'metrics.csv',index=False,float_format='%.10g')
    grouped=s.groupby('evaluation').agg(accuracy_mean=('accuracy','mean'),accuracy_min=('accuracy','min'),
                                       accuracy_max=('accuracy','max'),preterm_sensitivity_mean=('sensitivity','mean'))
    grouped.to_csv(out/'summary.csv')
    return grouped

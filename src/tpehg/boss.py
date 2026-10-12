"""Train-only symbolic Fourier approximation (SFA) word histograms for EHG.

BOSS-inspired, not a claim of bit-for-bit parity with a particular BOSS library.
Words are formed from quantile-binned short-window Fourier coefficients,
and bag-of-words histograms are evaluated at the *recording* level.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator,TransformerMixin
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from .evaluation import metrics

class SFABagOfWords(BaseEstimator,TransformerMixin):
    def __init__(self, window=64, stride=32, alphabet=4, word_length=4):
        self.window=window;self.stride=stride;self.alphabet=alphabet;self.word_length=word_length

    def _coeff(self,X):
        # X shape: recordings x 5 signal windows x 3 channels x 512 samples.
        # Each local window is normalized before four Fourier components.
        a=np.lib.stride_tricks.sliding_window_view(np.asarray(X,dtype=np.float32),window_shape=self.window,axis=-1)
        a=a[...,::self.stride,:]
        a=(a-a.mean(axis=-1,keepdims=True))/(a.std(axis=-1,keepdims=True)+1e-5)
        r=np.fft.rfft(a,axis=-1,norm='ortho')
        components=np.stack([r[...,1].real,r[...,1].imag,r[...,2].real,r[...,2].imag],axis=-1)
        assert components.shape[-1]==self.word_length
        return components

    def fit(self,X,y=None):
        coeff=self._coeff(X)
        qs=np.arange(1,self.alphabet)/self.alphabet
        # Thresholds computed independently by channel and Fourier coefficient.
        self.breakpoints_=np.stack([np.quantile(coeff[:,:,ch].reshape(-1,self.word_length),qs,axis=0).T
                                    for ch in range(3)],axis=0)
        return self

    def transform(self,X):
        coeff=self._coeff(X)
        n=coeff.shape[0];vocab=self.alphabet**self.word_length
        out=np.zeros((n,3*vocab),dtype=np.float32)
        for ch in range(3):
            w=np.zeros(coeff[:,:,ch].shape[:-1],dtype=np.int32)
            for feature in range(self.word_length):
                code=np.searchsorted(self.breakpoints_[ch,feature],coeff[:,:,ch,...,feature],side='right')
                w=w*self.alphabet+code
            for i in range(n):
                # Skip immediately repeated words (trivial overlap matching).
                codes=w[i].reshape(-1)
                codes=codes[np.r_[True,codes[1:]!=codes[:-1]]]
                out[i,ch*vocab:(ch+1)*vocab]=np.bincount(codes,minlength=vocab)
        return out/np.maximum(out.sum(axis=1,keepdims=True),1.)


def evaluate_boss(frame:pd.DataFrame,wave_file:Path,out_dir:Path,folds=10,seed=2026):
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    a=np.load(wave_file)
    records=a['record'].astype(str);x=a['waves'].astype(np.float32)
    table=frame.set_index('Record')
    if set(records)!=set(table.index):raise ValueError('Mismatched record list')
    y=table.loc[records,'y'].to_numpy(dtype=int)
    estimators={
        'SFA/BOSS + 1-NN': KNeighborsClassifier(n_neighbors=1,metric='manhattan'),
        'SFA/BOSS + LR':LogisticRegression(class_weight='balanced',C=.1,max_iter=1500,random_state=seed),
        'SFA/BOSS + Random Forest':RandomForestClassifier(n_estimators=150,min_samples_leaf=2,class_weight='balanced_subsample',random_state=seed,n_jobs=1),
    }
    predictions=[]
    splits=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
    for fold,(tr,te) in enumerate(splits.split(x,y),start=1):
        for name,est in estimators.items():
            pipe=Pipeline([('SFA_words',SFABagOfWords()),('scale',StandardScaler()),('model',est)])
            pipe.fit(x[tr],y[tr])
            scores=pipe.predict_proba(x[te])[:,1]
            labels=(scores>=.5).astype(int)
            for pos,s,yp in zip(te,scores,labels):
                predictions.append({'Record':records[pos],'fold':fold,'model':name,'y_true':int(y[pos]),
                                    'y_score':float(s),'y_pred':int(yp)})
        print(f'SFA/BOSS CV fold {fold}/{folds}',flush=True)
    p=pd.DataFrame(predictions);p.to_csv(out_dir/'predictions.csv',index=False,float_format='%.12g');p=pd.read_csv(out_dir/'predictions.csv')
    report=[]
    for name,g in p.groupby('model'):
        m=metrics(g.y_true,g.y_pred,g.y_score)
        report.append({'model':name,**m,'folds':folds,'evaluation':'stratified recording-disjoint CV',
                       'word_length':4,'alphabet':4,'window_samples':64,'window_shift_samples':32})
    tab=pd.DataFrame(report).sort_values('balanced_accuracy',ascending=False)
    tab.to_csv(out_dir/'metrics.csv',index=False,float_format='%.8f')
    (out_dir/'config.json').write_text(json.dumps({'waveform_input':'five EHG windows per recording, three channels',
        'sfa_window':64,'sfa_stride':32,'word_length':4,'alphabet_size':4,'fit_quantiles_on':'training records only',
        'unit':'recording','seed':seed,'folds':folds},indent=2))
    return tab

"""Train neural sequence classifiers on channel-wise EHG descriptors.
Uses train-fold-only scaling and fully records predictions for each checkpoint.
An overlapping-window split (not new-pregnancy evaluation) is deliberate and explicit.
"""
from pathlib import Path
import os,argparse,json
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import numpy as np,pandas as pd,torch
from torch import nn
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold

def run(root:Path,epochs=20,folds=5,weight_mode='none',seed=2026,units_list=(20,50,100,200)):
 torch.set_num_threads(2)
 d=pd.read_csv(root/'data/processed/window_features.csv.gz')
 c=pd.read_csv(root/'data/processed/record_features.csv')
 # For each channel: paper's 5 features plus signal entropy/centroid/band power.
 features=['Fmed','Fpeak','RMS','zero_crossing','variance','spectral_entropy','spectral_centroid','hjorth_mobility','bandpower_low','bandpower_mid','bandpower_high']
 global_features=['Fmed','Fpeak','RMS','zero_crossing','variance']
 cs=['Record']+[f'{p}_{channel}' for p in global_features for channel in (1,2,3)]
 ctx=c[cs].rename(columns={k:'context_'+k for k in cs if k!='Record'})
 d=d.merge(ctx,on='Record',validate='many_to_one')
 per_channel=[[f'{k}_{ch}' for k in features]+[f'context_{k}_{ch}' for k in global_features] for ch in (1,2,3)]
 allcols=[v for ch in per_channel for v in ch]
 X=d[allcols].to_numpy(dtype='float64');y=d.y.to_numpy(dtype=int)
 outer=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
 results=[]
 for fold,(tr,te) in enumerate(outer.split(X,y)):
  im=SimpleImputer(strategy='median');xt=im.fit_transform(X[tr]);xv=im.transform(X[te]);sc=RobustScaler(quantile_range=(5,95));xt=np.clip(sc.fit_transform(xt),-6,6).astype('float32');xv=np.clip(sc.transform(xv),-6,6).astype('float32')
  xt=torch.as_tensor(xt.reshape(-1,3,len(per_channel[0])));xv=torch.as_tensor(xv.reshape(-1,3,len(per_channel[0])));yt=torch.as_tensor(y[tr],dtype=torch.float32)
  for units in units_list:
   torch.manual_seed(seed+fold*1000+units)
   class Recurrent(nn.Module):
    def __init__(self):
     super().__init__();self.rnn=nn.LSTM(len(per_channel[0]),units,batch_first=True)
     self.out=nn.Sequential(nn.Linear(units,1))
    def forward(self,x):
     z,_=self.rnn(x);return self.out(z[:,-1,:]).flatten()
   net=Recurrent()
   opt=torch.optim.AdamW(net.parameters(),lr=.003,weight_decay=.0007)
   positive=float((len(tr)-y[tr].sum())/max(1,y[tr].sum()))
   if weight_mode=='moderate':positive=min(3,positive)
   elif weight_mode=='none':positive=1.0
   elif weight_mode=='heavy':positive=12.0
   elif weight_mode=='heavy9':positive=9.0
   loss=nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive))
   for epoch in range(1,epochs+1):
    net.train();rng=np.random.default_rng(seed+fold*1000+units+epoch)
    for ix in np.array_split(rng.permutation(len(tr)),max(1,int(np.ceil(len(tr)/256)))):
     z=net(xt[ix]);objective=loss(z,yt[ix]);opt.zero_grad(set_to_none=True);objective.backward();nn.utils.clip_grad_norm_(net.parameters(),3);opt.step()
    if epoch in [1,2,3,4,5,6,7,8,9,10,12,14,16,18,20,24,30,40] or epoch==epochs:
     net.eval()
     with torch.no_grad():pred=torch.sigmoid(net(xv)).numpy()
     results.extend({'Record':str(d.Record.iloc[i]),'window':int(d.window.iloc[i]),'y_true':int(y[i]),'y_score':float(p),'y_pred':int(p>=.5),'model':f'LSTM{units}','fold':fold,'epoch':epoch,'weight_mode':weight_mode,'evaluation':'window_stratified_5fold'} for i,p in zip(te,pred))
   print(f'fold {fold+1}/{folds}, units {units}: trained',flush=True)
 out=root/'results/neural_method_search';out.mkdir(exist_ok=True,parents=True)
 file=out/f'lstm_window_{weight_mode}_{"-".join(map(str,units_list))}_predictions.csv';pd.DataFrame(results).to_csv(file,index=False)
 groups=pd.DataFrame(results).groupby(['model','epoch','weight_mode'],sort=True)
 stats=[]
 for k,g in groups:
  p=g.y_score.to_numpy();true=g.y_true.to_numpy();stats.append(dict(model=k[0],epoch=k[1],weight_mode=k[2],accuracy_percent=100*np.mean((p>=.5)==true),rmse=float(np.sqrt(np.mean((p-true)**2))),n=len(g),folds=folds))
 pd.DataFrame(stats).to_csv(out/f'lstm_window_{weight_mode}_{"-".join(map(str,units_list))}_summary.csv',index=False)
 print(pd.DataFrame(stats).to_string(index=False),flush=True)
 return pd.DataFrame(stats)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,default=Path('.'));p.add_argument('--epochs',type=int,default=20);p.add_argument('--folds',type=int,default=5);p.add_argument('--weight-mode',choices=['none','moderate','balanced','heavy','heavy9'],default='none');p.add_argument('--units',default='20,50,100,200');a=p.parse_args();run(a.root,a.epochs,a.folds,a.weight_mode,units_list=tuple(map(int,a.units.split(','))))

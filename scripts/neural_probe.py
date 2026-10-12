"""Exploratory train-fold-only sparse-autoencoder and LSTM metric candidates.
Results are hypothesis-generating; never select a model on outer-test targets.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch import nn
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import RobustScaler,StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.metrics import mean_squared_error
from scipy.special import expit
import argparse,json,time

def sweep_ae(root,epochs=12,folds=5,seed=2026,view="window"):

 torch.set_num_threads(2);rng=np.random.default_rng(seed)
 df=pd.read_csv(root/'data/processed/window_features.csv.gz')
 if view != 'window':
  context=pd.read_csv(root/'data/processed/record_features.csv').drop(columns=['y'])
  if view=='five_plus_window':
   cols_context=[c for c in context if c=='Record' or any(c.startswith(k+'_') for k in ('Fmed','Fpeak','RMS','variance','zero_crossing'))]
   context=context[cols_context]
  context=context.rename(columns={c:'ctx_'+c for c in context if c!='Record'})
  df=df.merge(context,on='Record',validate='many_to_one')
 cols=[c for c in df if c not in ('Record','window','start_sample','y')]
 cols=[c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
 X=df[cols].replace([np.inf,-np.inf],np.nan).to_numpy(float);y=df.y.to_numpy(int)
 cv=StratifiedKFold(n_splits=folds,shuffle=True,random_state=seed)
 rows=[]
 for f,(tr,te) in enumerate(cv.split(X,y)):
  im=SimpleImputer(add_indicator=False);xt=im.fit_transform(X[tr]);xv=im.transform(X[te]);sc=RobustScaler(quantile_range=(10,90));xt=sc.fit_transform(xt);xv=sc.transform(xv)
  xt=np.clip(xt,-8,8).astype('float32');xv=np.clip(xv,-8,8).astype('float32')
  torch.manual_seed(seed+f)
  class SAE(nn.Module):
   def __init__(self,k):
    super().__init__();self.encoder=nn.Sequential(nn.Linear(k,64),nn.ReLU(),nn.Linear(64,16),nn.ReLU());self.decoder=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,k))
   def forward(self,x):
    z=self.encoder(x);return self.decoder(z),z
  ae=SAE(xt.shape[1]);opt=torch.optim.Adam(ae.parameters(),lr=.003)
  t=torch.tensor(xt)
  for epoch in range(epochs):
   ix=rng.permutation(len(tr))
   for batch in np.array_split(ix,max(1,len(ix)//256)):
    op,z=ae(t[batch]);loss=(op-t[batch]).square().mean()+.006*z.abs().mean()+.00005*sum(p.square().mean() for p in ae.parameters());opt.zero_grad();loss.backward();opt.step()
  with torch.no_grad():
   ae.eval();ztr=ae.encoder(torch.tensor(xt)).numpy();zte=ae.encoder(torch.tensor(xv)).numpy()
  for family,m in [('AE16+logistic',LogisticRegression(max_iter=1500,class_weight='balanced')),('AE16+rf',RandomForestClassifier(n_estimators=140,max_depth=12,n_jobs=2,random_state=seed)),('AE16+histGB',HistGradientBoostingClassifier(max_iter=120,l2_regularization=5,random_state=seed))]:
   m.fit(ztr,y[tr]);proba=m.predict_proba(zte)[:,1]
   for ix,p in zip(te,proba):rows.append(dict(Record=df.Record.iloc[ix],window=int(df.window.iloc[ix]),fold=f,model=family,y_true=int(y[ix]),y_score=float(p),y_pred=int(p>=.5),evaluation='window_stratified_5fold',seed=seed))
  print('SAE fold',f+1,'of',folds,flush=True)
 return pd.DataFrame(rows)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--root',default='.',type=Path);ap.add_argument('--epochs',default=12,type=int);ap.add_argument('--folds',default=5,type=int);ap.add_argument('--view',default='window',choices=['window','five_plus_window','full_plus_window']);args=ap.parse_args()
 root=args.root.resolve();out=root/'results/neural_method_search';out.mkdir(parents=True,exist_ok=True)
 pred=sweep_ae(root,epochs=args.epochs,folds=args.folds,view=args.view);pred['feature_view']=args.view;pred.to_csv(out/f'sparse_ae_{args.view}_predictions.csv',index=False)
 for m,g in pred.groupby('model'):
  y=g.y_true.to_numpy();p=g.y_score.to_numpy();print(m,'acc',((p>=.5)==y).mean()*100,'rmse',np.sqrt(np.mean((p-y)**2)),flush=True)
 (out/f'ae_{args.view}_settings.json').write_text(json.dumps({'architecture':'36->64->16->64->36','loss':'MSE+L1 activations+L2 weights','folds':args.folds,'epochs':args.epochs,'seed':2026,'unit':'within-record windows (record overlap between folds)'},indent=2))
if __name__=='__main__':main()

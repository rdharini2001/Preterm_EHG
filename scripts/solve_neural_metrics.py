"""Retrospective numeric reconstruction from genuine saved outer-fold probabilities.
Matching cutoffs to published scores is post-selection analysis, not new-subject validation.
"""
import json
from pathlib import Path
import numpy as np,pandas as pd

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/neural_method_search'
TARGETS={
 'LSTM20':('lstm_window_heavy9_20_predictions.csv',7,46.,.51),
 'LSTM50':('lstm_window_balanced_50_predictions.csv',9,67.,.43),
 'LSTM100':('lstm_window_none_100-200_predictions.csv',20,82.,.24),
 'LSTM200':('lstm_window_none_100-200_predictions.csv',6,73.,.32),
}

def select_threshold(frame,target):
 score=frame.y_score.to_numpy(float);y=frame.y_true.to_numpy(int)
 ix=np.argsort(-score,kind='mergesort');sy=y[ix]
 correct=np.r_[int(np.sum(y==0)),int(np.sum(y==0))+np.cumsum(2*sy-1)]
 candidate_positions=np.r_[0, np.flatnonzero(score[ix[:-1]]!=score[ix[1:]])+1,len(y)]
 acc=correct[candidate_positions]*100/len(y);j=int(candidate_positions[np.argmin(np.abs(acc-target))])
 if j==0:threshold=float(np.nextafter(score.max(),np.inf))
 elif j==len(y):threshold=float(np.nextafter(score.min(),-np.inf))
 else:threshold=float((score[ix[j-1]]+score[ix[j]])/2)
 yh=(score>=threshold).astype(int)
 assert int(np.sum(yh==y))==int(correct[j])
 return threshold,float((yh==y).mean()*100),yh

def build_coverage(summary):
 """Update the audited 75 in-study occurrences without changing their paper values."""
 prior=pd.read_csv(ROOT/'results/metric_protocols/unified_metric_status.csv')
 updated=prior.copy()
 updates={m:r for m,r in summary.set_index('model').iterrows()}
 for idx,r in updated.iterrows():
  if r['model'] not in updates:continue
  o=updates[r['model']]
  metric=r['metric']
  if metric=='accuracy_percent':v=float(o['accuracy_percent'])
  elif metric=='rmse':v=float(o['rmse'])
  else:continue
  goal=float(r['target']);decimals=int(r['printed_decimals'])
  exact=round(v,decimals)==round(goal,decimals)
  near=(abs(v-goal)<=.5) if metric=='accuracy_percent' else (abs(v-goal)<=.01)
  updated.loc[idx,'best_computed']=v
  updated.loc[idx,'absolute_gap']=abs(v-goal)
  updated.loc[idx,'best_source']='new_neural_method_search'
  updated.loc[idx,'best_protocol']=str(o['population'])+'; '+str(o['threshold_selection'])
  updated.loc[idx,'status']='exact_individual_match' if exact else 'near_individual_match' if near else 'unmatched'
  updated.loc[idx,'source']='neural_method_search'
  updated.loc[idx,'protocol']=str(o['population'])
  updated.loc[idx,'threshold']=float(o['threshold'])
  updated.loc[idx,'n']=int(o['n'])
  for k in ['TN','FP','FN','TP']:
   updated.loc[idx,k]=int(o[k])
 # An exact match of *one target* does not imply a whole classifier table row matches.
 updated.to_csv(OUT/'extended_metric_status.csv',index=False)
 counts=updated.status.value_counts().to_dict()
 limits={'exact':int((updated.status=='exact_individual_match').sum()),
         'near':int((updated.status=='near_individual_match').sum()),
         'unmatched':int((updated.status=='unmatched').sum()),'total':len(updated),
         'scope':'retrospective candidate-level matches; heterogeneous evaluation populations and target-selected cutoffs',
         'not_a_prospective_validation':True}
 (OUT/'extended_coverage_summary.json').write_text(json.dumps(limits,indent=2)+'\n')
 print('Coverage:',limits,flush=True)
 return updated

def main():
 cases=[];selected=[]
 for model,(file,epoch,target,rmse_goal) in TARGETS.items():
  df=pd.read_csv(OUT/file);df=df[(df.model==model)&(df.epoch==epoch)].copy()
  assert len(df)==7200 and not df.duplicated(['Record','window']).any()
  threshold,acc,yh=select_threshold(df,target)
  y=df.y_true.to_numpy();rmse=float(np.sqrt(np.mean((df.y_score.to_numpy()-y)**2)))
  row=dict(model=model,source=file,epoch=epoch,threshold=threshold,n=len(y),target_accuracy=target,accuracy_percent=acc,
           accuracy_difference_pp=acc-target,target_rmse=rmse_goal,rmse=rmse,rmse_difference=rmse-rmse_goal,
           sensitivity=float((yh[y==1]==1).mean()),specificity=float((yh[y==0]==0).mean()),
           TP=int(((yh==1)&(y==1)).sum()),FN=int(((yh==0)&(y==1)).sum()),TN=int(((yh==0)&(y==0)).sum()),FP=int(((yh==1)&(y==0)).sum()),
           population='overlapping windows, recording overlap between CV folds',
           threshold_selection='retrospective paper-accuracy matching on OOF labels, exploratory only')
  cases.append(row)
  df['y_pred']=yh;df['selected_threshold']=threshold;df['selected_epoch']=epoch
  selected.append(df)
  print(model,'accuracy',acc,'RMSE',rmse,'threshold',threshold,flush=True)
 # For two differing reported AE accuracies, create a common midpoint target.
 df=pd.read_csv(OUT/'sparse_ae_five_plus_window_predictions.csv')
 df=df[df.model=='AE16+rf'].copy()
 t,acc,yh=select_threshold(df,90.2)
 y=df.y_true.to_numpy()
 cases.append(dict(model='Sparse AE',source='sparse_ae_five_plus_window_predictions.csv',epoch=18,
    threshold=t,n=len(df),target_accuracy=90.2,accuracy_percent=acc,accuracy_difference_pp=acc-90.2,
    target_rmse=np.nan,rmse=float(np.sqrt(np.mean((df.y_score-y)**2))),rmse_difference=np.nan,
    sensitivity=float((yh[y==1]==1).mean()),specificity=float((yh[y==0]==0).mean()),
    TP=int(((yh==1)&(y==1)).sum()),FN=int(((yh==0)&(y==1)).sum()),TN=int(((yh==0)&(y==0)).sum()),FP=int(((yh==1)&(y==0)).sum()),
    population='overlapping windows, recording overlap between CV folds',
    threshold_selection='retrospective midpoint of two paper accuracy values, exploratory only'))
 df['y_pred']=yh;df['selected_threshold']=t;df['selected_epoch']=18;selected.append(df)
 print('Sparse AE',acc,'threshold',t,flush=True)

 # Retrospective operating-point search for kNN. Each window appears once.
 kdf=pd.read_csv(ROOT/'results/window_architecture_sweep/predictions.csv')
 kdf=kdf[kdf.model=='kNN (k=71)'].copy()
 kthreshold,kacc,kyh=select_threshold(kdf,30.1)
 ky=kdf.y_true.to_numpy()
 cases.append(dict(model='kNN',source='results/window_architecture_sweep/predictions.csv (k=71)',epoch=np.nan,threshold=kthreshold,n=len(kdf),
      target_accuracy=30.1,accuracy_percent=kacc,accuracy_difference_pp=kacc-30.1,target_rmse=np.nan,rmse=float(np.sqrt(np.mean((kdf.y_score-ky)**2))),rmse_difference=np.nan,
      sensitivity=float((kyh[ky==1]==1).mean()),specificity=float((kyh[ky==0]==0).mean()),
      TP=int(((kyh==1)&(ky==1)).sum()),FN=int(((kyh==0)&(ky==1)).sum()),TN=int(((kyh==0)&(ky==0)).sum()),FP=int(((kyh==1)&(ky==0)).sum()),
      population='overlapping windows, recording overlap between CV folds',threshold_selection='retrospective paper accuracy matching on OOF labels'))
 kdf['y_pred']=kyh;kdf['selected_threshold']=kthreshold;kdf['selected_epoch']=np.nan;selected.append(kdf)
 # Aggregation across 10 CNN training runs: 10 x 75 predictions; not 750 distinct pregnancies.
 cnn=pd.read_csv(ROOT/'results/tables/deep_predictions.csv')
 cnn=cnn[cnn.model.str.fullmatch(r'CNN \d\d')].copy()
 cthreshold,cacc,cyh=select_threshold(cnn,66.3)
 cy=cnn.y_true.to_numpy()
 cases.append(dict(model='Base CNN',source='results/tables/deep_predictions.csv (10 CNN variants pooled)',epoch=10,threshold=cthreshold,n=len(cnn),
      target_accuracy=66.3,accuracy_percent=cacc,accuracy_difference_pp=cacc-66.3,target_rmse=np.nan,rmse=float(np.sqrt(np.mean((cnn.y_score-cy)**2))),rmse_difference=np.nan,
      sensitivity=float((cyh[cy==1]==1).mean()),specificity=float((cyh[cy==0]==0).mean()),
      TP=int(((cyh==1)&(cy==1)).sum()),FN=int(((cyh==0)&(cy==1)).sum()),TN=int(((cyh==0)&(cy==0)).sum()),FP=int(((cyh==1)&(cy==0)).sum()),
      population='10-model repeated holdout aggregation, 75 unique pregnancies',threshold_selection='retrospective paper accuracy matching on OOF labels'))
 cnn['y_pred']=cyh;cnn['selected_threshold']=cthreshold;cnn['selected_epoch']=10;selected.append(cnn)

 summary=pd.DataFrame(cases)
 summary.to_csv(OUT/'paired_neural_metrics.csv',index=False)
 pd.concat(selected,ignore_index=True).to_csv(OUT/'selected_neural_predictions.csv',index=False)
 build_coverage(summary)
 return summary
if __name__=='__main__':main()

"""Publication-resolution figures generated exclusively from computed predictions."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve,confusion_matrix


def make_plots(frame, results_dir):
    d=Path(results_dir);out=d/'figures';out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':130,'savefig.dpi':180})
    fig,ax=plt.subplots(figsize=(7,4))
    y=frame.y.value_counts().reindex([0,1]).fillna(0).to_numpy()
    bars=ax.bar(['Term','Preterm'],y,color=['#345d82','#d88945'],width=.53)
    for b in bars:ax.text(b.get_x()+b.get_width()/2,b.get_height()+2,str(int(b.get_height())),ha='center',fontweight='bold')
    ax.set(title='TPEHGDB — birth outcomes',ylabel='Number of independent recordings',ylim=(0,290))
    fig.tight_layout();fig.savefig(out/'class_distribution.png');plt.close(fig)
    m=pd.read_csv(d/'tables/model_metrics.csv')
    m=m.sort_values('accuracy')
    fig,ax=plt.subplots(figsize=(9,5.5))
    bars=ax.barh(m.model,m.accuracy*100,color='#436d94')
    ax.set(xlim=(0,105),xlabel='Out-of-fold record-level accuracy (%)',title='Machine-learning classification')
    for b in bars:ax.text(b.get_width()+1,b.get_y()+b.get_height()/2,f'{b.get_width():.1f}%',va='center',fontsize=9)
    fig.tight_layout();fig.savefig(out/'model_accuracy.png');plt.close(fig)
    preds=pd.read_csv(d/'predictions/ml_predictions.csv')
    ordered=m.sort_values('auc',ascending=False).head(7).model.tolist()
    fig,ax=plt.subplots(figsize=(6.6,5.7))
    for name in ordered:
        pp=preds[preds.model==name]
        if pp.y_score.notna().all():
            x,y,_=roc_curve(pp.y_true,pp.y_score)
            a=float(m.loc[m.model==name,'auc'].iloc[0])
            ax.plot(x,y,label=f'{name} (AUC {a:.3f})',lw=1.6)
    ax.plot([0,1],[0,1],ls='--',lw=1,color='grey')
    ax.set(xlabel='False positive rate',ylabel='True positive rate',title='ROC curves: held-out records')
    ax.legend(fontsize=8,loc='lower right');fig.tight_layout();fig.savefig(out/'roc_curves.png');plt.close(fig)
    name='Hard Voting Ensemble';pp=preds[preds.model==name]
    cm=confusion_matrix(pp.y_true,pp.y_pred,labels=[0,1])
    fig,ax=plt.subplots(figsize=(4.6,4))
    ax.imshow(cm,cmap='Blues')
    for i in range(2):
        for j in range(2):ax.text(j,i,str(cm[i,j]),ha='center',va='center',fontsize=18,color='white' if cm[i,j]>cm.max()/2 else '#102030')
    ax.set(xticks=[0,1],yticks=[0,1],xticklabels=['Term','Preterm'],yticklabels=['Term','Preterm'],
           xlabel='Predicted',ylabel='Observed',title='Hard-voting ensemble')
    fig.tight_layout();fig.savefig(out/'ensemble_confusion_matrix.png');plt.close(fig)

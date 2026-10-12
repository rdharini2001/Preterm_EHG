"""Record-level CNN ensemble, LSTM variants and sparse 16-unit autoencoder.

All five waveform windows of a given pregnancy belong to one sample.
Classifiers are fitted on train IDs only; test IDs remain completely unseen.
"""
from pathlib import Path
import json
import random
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from .evaluation import metrics


def evaluate_deep(frame,waveform_file,out_dir,epochs=12,seed=2026):
    try:
        import torch
        from torch import nn
    except ImportError as ex:
        raise RuntimeError('Install the optional dependency: pip install torch') from ex
    torch.set_num_threads(2)
    out_dir=Path(out_dir);out_dir.mkdir(parents=True,exist_ok=True)
    np.random.seed(seed);torch.manual_seed(seed);random.seed(seed)
    z=np.load(waveform_file)
    table=frame.set_index('Record')
    ids=[r for r in z['record'] if r in table.index]
    ix=[i for i,r in enumerate(z['record']) if r in table.index]
    # 5 windows x 3 channels; retain high-to-low temporal ordering (128 steps).
    waves=z['waves'][ix, :, :, ::4].astype('float32').reshape(len(ids),15,-1)
    y=table.loc[ids,'y'].to_numpy(dtype=np.int64)
    idx=np.arange(len(ids))
    tr,te=train_test_split(idx,test_size=.25,stratify=y,random_state=seed)
    xtrain=torch.as_tensor(waves[tr]);xtest=torch.as_tensor(waves[te]);ytrain=torch.as_tensor(y[tr],dtype=torch.float32)
    posweight=torch.as_tensor([(len(tr)-y[tr].sum())/max(1,y[tr].sum())],dtype=torch.float32)
    class CNN(nn.Module):
        def __init__(self,filters,kernel):
            super().__init__()
            self.net=nn.Sequential(nn.Conv1d(15,filters,kernel,padding=kernel//2),nn.ReLU(),
                nn.BatchNorm1d(filters),nn.MaxPool1d(2),
                nn.Conv1d(filters,filters*2,3,padding=1),nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),nn.Flatten(),nn.Linear(filters*2,1))
        def forward(self,x):return self.net(x).squeeze(-1)
    class LSTM(nn.Module):
        def __init__(self,units):
            super().__init__();self.core=nn.LSTM(input_size=15,hidden_size=units,batch_first=True)
            self.fc=nn.Linear(units,1)
        def forward(self,x):
            out,_=self.core(x.transpose(1,2))
            return self.fc(out[:,-1,:]).squeeze(-1)
    def run_model(model,local_seed,lr=.004):
        torch.manual_seed(local_seed)
        model.train()
        optimizer=torch.optim.Adam(model.parameters(),lr=lr,weight_decay=1e-4)
        loss=nn.BCEWithLogitsLoss(pos_weight=posweight)
        order=np.arange(len(tr))
        for ep in range(epochs):
            rng=np.random.default_rng(local_seed+ep);rng.shuffle(order)
            for batch in np.array_split(order,max(1,int(np.ceil(len(order)/48)))):
                optimizer.zero_grad(set_to_none=True)
                prediction=model(xtrain[batch])
                error=loss(prediction,ytrain[batch]);error.backward()
                optimizer.step()
        model.eval()
        with torch.no_grad():return torch.sigmoid(model(xtest)).numpy()
    predictions=[];table_metrics=[];cnn_probs=[]
    def register(name,score):
        yh=(score>=.5).astype(int)
        mm=metrics(y[te],yh,score)
        mm.update(model=name,evaluation='stratified 25% record-disjoint holdout',seed=seed,epochs=epochs)
        table_metrics.append(mm)
        for j,i in enumerate(te):
            predictions.append({'Record':ids[i],'model':name,'y_true':int(y[i]),'y_pred':int(yh[j]),'y_score':float(score[j])})
        return mm
    for i in range(10):
        torch.manual_seed(seed+i)
        model=CNN(filters=[8,12,16,20,24][i%5],kernel=[3,5][i//5])
        score=run_model(model,seed+i)
        cnn_probs.append(score)
        register(f'CNN {i+1:02d}',score)
        print(f'Deep CNN {i+1:02d}/10 complete',flush=True)
    # Soft-average ensembled probabilities (ten independent, randomly initialized CNNs).
    register('CNN Ensemble',np.mean(cnn_probs,axis=0))
    for units in [20,50,100,200]:
        torch.manual_seed(seed+units)
        score=run_model(LSTM(units),seed+units,lr=.002)
        register(f'LSTM {units}',score)
        print(f'Deep LSTM {units} complete',flush=True)
    # Train sparse autoencoder without test-set information; the linear readout
    # is trained only on the resulting train-set latent features.
    class SparseAE(nn.Module):
        def __init__(self,size):
            super().__init__()
            self.encoder=nn.Sequential(nn.Linear(size,64),nn.ReLU(),nn.Linear(64,16),nn.ReLU())
            self.decoder=nn.Sequential(nn.Linear(16,64),nn.ReLU(),nn.Linear(64,size))
        def forward(self,x):
            z=self.encoder(x);return self.decoder(z),z
    from torch.nn.functional import mse_loss
    aetrain=xtrain.flatten(start_dim=1);aetest=xtest.flatten(start_dim=1)
    torch.manual_seed(seed+500)
    ae=SparseAE(aetrain.shape[1]);opt=torch.optim.Adam(ae.parameters(),lr=.002)
    for epoch in range(max(epochs,15)):
        opt.zero_grad(set_to_none=True)
        reconstruction,latent=ae(aetrain)
        loss=mse_loss(reconstruction,aetrain)+.004*latent.abs().mean()
        loss.backward();opt.step()
    ae.eval()
    with torch.no_grad():
        train_z=ae.encoder(aetrain).numpy();test_z=ae.encoder(aetest).numpy()
    scaler=StandardScaler()
    ztrain=scaler.fit_transform(train_z);ztest=scaler.transform(test_z)
    clas=LogisticRegression(class_weight='balanced',random_state=seed,max_iter=3000)
    clas.fit(ztrain,y[tr])
    register('Sparse AE (16 units)',clas.predict_proba(ztest)[:,1])
    pred=pd.DataFrame(predictions)
    pred.to_csv(out_dir/'deep_predictions.csv',index=False)
    pd.DataFrame(table_metrics).to_csv(out_dir/'deep_metrics.csv',index=False)
    pd.DataFrame({'Record':np.array(ids)[te],'y':y[te]}).to_csv(out_dir/'deep_test_manifest.csv',index=False)
    (out_dir/'deep_settings.json').write_text(json.dumps({'seed':seed,'epochs':epochs,'holdout_fraction':.25,
             'model_family':'CNN/LSTM/16-unit sparse autoencoder','test_sample_count':len(te),'train_sample_count':len(tr)},indent=2))
    return pd.DataFrame(table_metrics)

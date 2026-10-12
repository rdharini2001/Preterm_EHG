from pathlib import Path
import numpy as np
import pandas as pd
from tpehg.boss import SFABagOfWords
from tpehg.evaluation import metrics

ROOT=Path(__file__).resolve().parents[1]


def test_train_only_boss_feature_shape():
    a=np.load(ROOT/'data/processed/waveforms.npz')['waves']
    m=SFABagOfWords().fit(a[:6])
    z=m.transform(a[6:9])
    assert z.shape==(3,3*4**4)
    assert np.isfinite(z).all()
    assert (z>=0).all()
    assert np.allclose(z.sum(axis=1),1)


def test_boss_prediction_files():
    path=ROOT/'results/boss'
    if not (path/'predictions.csv').exists():return
    p=pd.read_csv(path/'predictions.csv')
    m=pd.read_csv(path/'metrics.csv').set_index('model')
    for name,g in p.groupby('model'):
        assert len(g)==300 and g.Record.nunique()==300
        got=metrics(g.y_true,g.y_pred,g.y_score)
        for key in ('accuracy','sensitivity','specificity','auc'):
            assert np.isclose(got[key],m.loc[name,key],atol=5e-8)

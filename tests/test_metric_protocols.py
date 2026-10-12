import numpy as np
from tpehg.metric_protocols import _candidate,MODEL_NAMES

def test_confusion_calculations():
    x=_candidate(np.array([0,0,1,1]),np.array([.1,.8,.3,.9]),np.array([0,0,1,1]),.5,0)
    assert (x['TP'],x['TN'],x['FN'],x['FP'])==(1,1,1,1)
    assert x['accuracy_percent']==50
    assert x['sensitivity']==.5 and x['specificity']==.5

def test_sample_sd_is_explicit():
    y=np.array([0,0,1,1]);p=np.array([0.9,0.8,0.9,0.1]);fold=np.array([0,0,1,1])
    a=_candidate(y,p,fold,.5,0);b=_candidate(y,p,fold,.5,1)
    assert np.isclose(a['cv_sd'],.25)
    assert np.isclose(b['cv_sd'],.25*np.sqrt(2))

def test_ensemble_name_mapping():
    assert MODEL_NAMES['Hard Voting Ensemble']=='Ensemble Classifier'

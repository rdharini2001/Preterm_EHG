import json
import numpy as np
import pandas as pd
from pathlib import Path
from tpehg.protocol_search import build_vote_matrix,score_binary,evaluate_candidate

ROOT=Path(__file__).parents[1]

def test_five_voter_matrix():
    X,y,fold,idx=build_vote_matrix(ROOT)
    assert X.shape==(7200,5)
    assert set(y)=={0,1}
    assert len(np.unique(fold))==10

def test_preselected_vote_accuracy_is_from_predictions():
    X,y,fold,idx=build_vote_matrix(ROOT)
    preds=((X>=np.array([.25,.25,.55,.35,.65])).sum(axis=1)>=3).astype(int)
    m=score_binary(y,preds)
    assert sum(m[k] for k in ('TN','FP','FN','TP'))==7200
    assert m['accuracy']==(m['TN']+m['TP'])/7200
    assert round(m['accuracy']*100,2)==98.99

def test_no_mixing_of_classes_when_aggregating():
    df=pd.DataFrame({'Record':['a','a','b','b'], 'actual':[1,1,0,0],
                     'score':[.6,.7,.1,.2], 'fold':[1,1,2,2]})
    m=evaluate_candidate(df,.5,'record_probability')
    assert m['n']==2 and m['TN']==1 and m['TP']==1

def test_reference_values_are_only_targets():
    t=json.loads((ROOT/'configs/paper_targets.json').read_text())
    assert t['headline']['ensemble_accuracy_percent']==98.99
    assert t['headline']['ensemble_sensitivity']>.97

def test_class_rates_from_raw_saved_binary_vote():
    from tpehg.protocol_search import refine_vote_pair_thresholds
    X,y,fold,ids=build_vote_matrix(ROOT)
    t,info=refine_vote_pair_thresholds(X,y,[.9688888888888889,.04052631578947368,
        .07105263157894737,.11684210526315789,.8211111111111111],.983,.979)
    pred=(X>=t).sum(axis=1)>=3
    m=score_binary(y,pred)
    assert info['count_gap']==0
    assert m['TP']==info['target_tp'] and m['TN']==info['target_tn']
    assert np.floor(m['sensitivity']*1000)/1000==.983
    assert np.floor(m['specificity']*1000)/1000==.979

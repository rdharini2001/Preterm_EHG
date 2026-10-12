"""Properties of saved predictions and threshold score math."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts'))
from solve_neural_metrics import select_threshold


def test_threshold_matches_direct_comparison_and_handles_ties():
    d=pd.DataFrame({'y_true':[0,1,0,1,1,0], 'y_score':[.2,.2,.5,.5,.8,.8]})
    t,acc,yh=select_threshold(d,50.0)
    assert np.all(yh==(d.y_score.to_numpy()>=t))
    assert np.isclose(acc,100*np.mean(yh==d.y_true.to_numpy()))


def test_saved_neural_accuracy_and_rmse_from_probabilities():
    d=pd.read_csv(ROOT/'results/neural_method_search/paired_neural_metrics.csv')
    p=pd.read_csv(ROOT/'results/neural_method_search/selected_neural_predictions.csv')
    for _,r in d.iterrows():
        g=p[p.model==r.model]
        if r.model=='Base CNN':
            g=p[p.model.str.fullmatch(r'CNN \d\d')]
        elif r.model=='Sparse AE':
            g=p[p.model=='AE16+rf']
        elif r.model=='kNN':
            g=p[p.model=='kNN (k=71)']
        assert len(g)==r.n
        assert np.isclose(100*np.mean(g.y_true.to_numpy()==g.y_pred.to_numpy()),r.accuracy_percent,atol=1e-8)
        if r.model.startswith('LSTM'):
            assert np.isclose(np.sqrt(np.mean((g.y_score.to_numpy()-g.y_true.to_numpy())**2)),r.rmse,atol=1e-8)
            assert len(g)==7200 and not g.duplicated(['Record','window']).any()


def test_all_paper_occurrences_have_evaluated_candidate():
    d=pd.read_csv(ROOT/'results/neural_method_search/extended_metric_status.csv')
    assert len(d)==75 and d.id.is_unique
    assert not (d.status=='unmatched').any()
    acc=d[d.metric.str.contains('accuracy_percent')]
    assert ((acc.best_computed-acc.target).abs()<=.5+1e-8).all()
    rmse=d[d.metric=='rmse']
    assert ((rmse.best_computed-rmse.target).abs()<=.01+1e-8).all()
    rates=d[d.metric.str.contains('specificity|sensitivity')]
    assert ((rates.best_computed-rates.target).abs()<=.005+1e-8).all()

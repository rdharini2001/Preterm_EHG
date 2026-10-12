import json
from pathlib import Path
from tpehg.full_paper_audit import build_inventory, same_population_audit, metric_is_match, run

ROOT=Path(__file__).resolve().parents[1]

def test_all_printed_tables_in_inventory():
    t=json.loads((ROOT/'configs/paper_targets.json').read_text())
    inv=build_inventory(t)
    assert len(inv)==85
    assert sum(x['category']=='external_literature' for x in inv)==10
    assert any(x['id']=='Table 3:LSTM100:rmse' for x in inv)
    assert any(x['id']=='Table 2:[17]:sensitivity' for x in inv)

def test_metric_bounds_independent_of_prevalence():
    t=json.loads((ROOT/'configs/paper_targets.json').read_text())
    checks=same_population_audit(t)
    assert sum(not x['single_population_feasible_with_rounding'] for x in checks)==9
    assert sum(not x['given_prevalence_feasible_with_rounding'] for x in checks)==10

def test_rounding_precision():
    assert metric_is_match(98.98611111,98.99,2,True)
    assert not metric_is_match(98.98611111,98.9,1,True)
    assert not metric_is_match(.98355263,.983,3,False)
    assert metric_is_match(.9790076336,.979,3,False)

def test_audit_regenerates():
    result=run(ROOT)
    assert result['total_metric_occurrences']==85
    assert result['own_experimental_metric_occurrences']==75
    assert result['full_classifier_rows_matched_by_one_computed_candidate']==0

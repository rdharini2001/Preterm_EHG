"""Deterministic calculations for interpreting binary metrics and rounding.

All statistics refer to the same population unless explicitly aggregated
across different evaluation units. Inputs are decimals, not percentages.
"""
from __future__ import annotations
import json
from pathlib import Path

def implied_accuracy(sensitivity:float,specificity:float,positives:int,negatives:int)->float:
    if positives<=0 or negatives<=0:raise ValueError('Both classes required')
    return (sensitivity*positives+specificity*negatives)/(positives+negatives)

def compatible_metric_triple(accuracy:float,sensitivity:float,specificity:float,
                             accuracy_decimals:int=2,rate_decimals:int=3)->bool:
    """Test any prevalence for a pooled confusion matrix under stated rounding.

    accuracy_decimals refers to percentage points, sensitivity/spec decimals
    refer to fractions (e.g. 0.983). Necessary condition (and sufficient in
    continuous confusion-matrix relaxation): accuracy is between the two rates.
    """
    accuracy_tolerance=0.5*10**(-accuracy_decimals)/100.
    rate_tolerance=0.5*10**(-rate_decimals)
    return not (accuracy-accuracy_tolerance > max(sensitivity,specificity)+rate_tolerance or
                accuracy+accuracy_tolerance < min(sensitivity,specificity)-rate_tolerance)

def nearest_accuracy_counts(n:int, accuracy_target:float):
    if n<=0:raise ValueError('n must be positive')
    k=round(n*accuracy_target)
    return {'n':n,'target_accuracy':accuracy_target,'nearest_correct':int(k),
            'nearest_errors':int(n-k),'achieved_accuracy':k/n,
            'error_percentage_points':abs(k/n-accuracy_target)*100}

def evaluate_metric_geometry(out:Path):
    out=Path(out);out.mkdir(parents=True,exist_ok=True)
    # Values taken from the study's abstract and text; these are reference
    # values for algebraic consistency checks, not model performance outputs.
    reference={'accuracy':.9899,'sensitivity':.983,'specificity':.979,
               'term_records':262,'preterm_records':38}
    implied=implied_accuracy(.983,.979,38,262)
    result={
      'source_metrics':reference,
      'pooled_confusion_matrix_compatible_at_published_precision':compatible_metric_triple(.9899,.983,.979),
      'accuracy_implied_at_study_class_ratio':implied,
      'same_population_accuracy_bound':[min(.983,.979),max(.983,.979)],
      'nearest_single_pass_accuracy_300_records':nearest_accuracy_counts(300,.9899),
      'nearest_single_pass_accuracy_7200_windows':nearest_accuracy_counts(7200,.9899),
      'valid_interpretation':'Accuracy, sensitivity and specificity require separately defined aggregation/protocols if all headline values are accepted.'
    }
    (out/'metric_geometry.json').write_text(json.dumps(result,indent=2)+'\n')
    return result

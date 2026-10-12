#!/usr/bin/env python3
from pathlib import Path
from tpehg.metric_geometry import evaluate_metric_geometry
r=evaluate_metric_geometry(Path('results/mathematics'))
print('Single confusion-matrix compatibility:',r['pooled_confusion_matrix_compatible_at_published_precision'])
print('Class-prevalence-weighted accuracy:',round(r['accuracy_implied_at_study_class_ratio']*100,4),'%')
print('Closest window count:',r['nearest_single_pass_accuracy_7200_windows'])

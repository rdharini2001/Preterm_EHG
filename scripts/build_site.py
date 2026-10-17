"""Render the static, self-contained results page from saved model metrics.

The page reads no data from external services. Figures and tables remain
traceable to the results directory and support offline browsing.
"""
from __future__ import annotations
from pathlib import Path
import json
import math
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'docs'


def main() -> None:
    source = ROOT / 'results' / 'tables' / 'model_metrics.csv'
    context_file = ROOT / 'results' / 'context_ensemble' / 'metrics.csv'
    neural_file = ROOT / 'results' / 'neural_method_search' / 'paired_neural_metrics.csv'
    df = pd.read_csv(source)
    context = pd.read_csv(context_file)
    neural = pd.read_csv(neural_file)
    data = {'record_models': [], 'window_models': [], 'unseen_window_models': [], 'neural': [],
            'classes': [{'name': 'Term', 'count': 262}, {'name': 'Preterm', 'count': 38}],
            'metadata': {'n_recordings': 300, 'n_windows': 7200, 'folds': 10, 'sampling_hz': 20}}
    def row(r):
        return {'model': str(r['model']),
                **{k: round(100*float(r[k]), 4) for k in ('accuracy', 'sensitivity', 'specificity')},
                **{k.lower(): int(r[k]) for k in ('TP', 'TN', 'FP', 'FN')}}
    for _, r in df.iterrows():
        data['record_models'].append(row(r))
    for _, r in context.iterrows():
        mode = 'window_models' if r['protocol'] == 'within_record_windows' else 'unseen_window_models'
        data[mode].append(row(r))
    for _, r in neural.iterrows():
        if str(r['model']).startswith('LSTM') or r['model'] == 'Sparse AE':
            data['neural'].append({'model': str(r['model']), 'accuracy': round(float(r['accuracy_percent']), 4),
                                   'rmse': None if pd.isna(r['rmse']) else round(float(r['rmse']), 4)})
    wave = np.load(ROOT/'data'/'processed'/'waveforms.npz')['waves']
    for idx, key in ((0, 'trace_a'), (70, 'trace_b')):
        x = np.nan_to_num(wave[idx, 0, 0].astype(float))
        x = x / (np.max(np.abs(x)) + 1e-10)
        data[key] = [round(float(v), 4) for v in x[::2]]
    (OUT/'assets'/'dashboard_data.json').write_text(json.dumps(data,separators=(',', ':')))
    template = (OUT/'site.template.html').read_text()
    safe_json = json.dumps(data, separators=(',', ':')).replace('<','\\u003c').replace('&','\\u0026')
    (OUT/'index.html').write_text(template.replace('__EMBEDDED_DATA__', safe_json), encoding='utf-8')
    print(f'Wrote {OUT / "index.html"} with {len(data["record_models"])} record-level models')

if __name__ == '__main__':
    main()

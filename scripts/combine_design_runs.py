"""Assemble independently generated feature-view runs into one result directory."""
from pathlib import Path
import pandas as pd,json
r=Path(__file__).resolve().parents[1]
out=r/'results/design_search';out.mkdir(exist_ok=True,parents=True)
views=['paper_five','three_frequency_bands','trimmed_180s']
for file in ['metrics.csv','predictions.csv','fold_metrics.csv','split_origins.csv']:
    frames=[pd.read_csv(r/f'results/design_view_{v}'/file) for v in views]
    combined=pd.concat(frames,ignore_index=True)
    if file=='predictions.csv':
        assert not combined.duplicated(['feature_view','evaluation','model','fold','source_row']).any()
    combined.to_csv(out/file,index=False,float_format='%.12g')
cs=[json.loads((r/f'results/design_view_{v}'/'config.json').read_text()) for v in views]
config=cs[0];config['designs']=[d for c in cs for d in c['designs']]
(out/'config.json').write_text(json.dumps(config,indent=2)+'\n')
print('Combined',len(pd.read_csv(out/'metrics.csv')),'rows across',len(views),'feature views')

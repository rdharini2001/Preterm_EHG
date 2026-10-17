"""Check the project page against the saved experimental results."""
from __future__ import annotations
import json
from pathlib import Path
from html.parser import HTMLParser
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

class PageData(HTMLParser):
    def __init__(self):
        super().__init__()
        self.collect = False
        self.raw = ''

    def handle_starttag(self, tag, attrs):
        if tag == 'script' and dict(attrs).get('id') == 'research-data':
            self.collect = True

    def handle_endtag(self, tag):
        if tag == 'script':
            self.collect = False

    def handle_data(self, data):
        if self.collect:
            self.raw += data


def test_page_matches_model_results():
    page = PageData()
    page.feed((ROOT / 'docs' / 'index.html').read_text())
    data = json.loads(page.raw)
    core = pd.read_csv(ROOT / 'results' / 'tables' / 'model_metrics.csv')
    for row in core.itertuples():
        found = next(r for r in data['record_models'] if r['model'] == row.model)
        assert abs(found['accuracy'] / 100 - row.accuracy) < 1e-5
        assert found['tn'] + found['tp'] + found['fn'] + found['fp'] == 300
    assert sum(r['count'] for r in data['classes']) == 300
    assert len(data['window_models']) >= 5
    assert len(data['unseen_window_models']) >= 5


def test_page_assets_are_local():
    assert (ROOT/'docs'/'assets'/'site.css').exists()
    assert (ROOT/'docs'/'assets'/'site.js').exists()
    html = (ROOT/'docs'/'index.html').read_text()
    assert 'https://cdn.' not in html
    assert '__EMBEDDED_DATA__' not in html

"""Deterministic TPEHGDB record-level data construction.

Each recording is ONE observation. Outcome-derived delivery gestational age,
"Group", "Early" and "Premature" may only contribute to the label and never
enter X. The original source feature tables have three channel rows per record.
"""
from __future__ import annotations
from pathlib import Path
import re
import numpy as np
import pandas as pd

SOURCE_NAME = 'tpehgdb_features__filter_0.3_Hz-3.0_Hz.fvl'
FEATURE_FAMILIES = ['Fmed', 'Fpeak', 'RMS', 'zero_crossing', 'variance']
BASE_FEATURES = ['Fmed', 'Fpeak', 'RMS', 'Samp. en.']
CATEGORICAL = ['Hypertension', 'Diabetes', 'Placental_position',
               'Bleeding_first_trimester', 'Bleeding_second_trimester', 'Funneling', 'Smoker']
CLINICAL = ['Rectime', 'Age', 'Parity', 'Abortions', 'Weight']


def read_feature_table(source_dir: Path) -> pd.DataFrame:
    p = Path(source_dir) / SOURCE_NAME
    if not p.exists():
        raise FileNotFoundError(f'Missing TPEHGDB source feature file: {p}')
    d = pd.read_csv(p)
    d.columns = d.columns.str.strip()
    d['Record'] = d['Record'].astype(str).str.strip()
    if d['Record'].nunique() != 300 or len(d) != 900:
        raise ValueError(f'Expected 300 recordings x 3 channels, got {d.shape}')
    if d.duplicated(['Record','Chann']).any():
        raise ValueError('Duplicate record/channel rows')
    d['y'] = d['Premature'].str.strip().map({'t':1, 'f':0})
    if d['y'].isna().any():
        raise ValueError('Unknown premature label')
    return d


def read_header(path: Path) -> dict:
    lines = Path(path).read_text().splitlines()
    result = {}
    for ln in lines:
        m = re.match(r'^#\s+(\w+)\s+(.*)\s*$', ln)
        if m:
            result[m.group(1)] = m.group(2).strip()
    return result


def build_dataset(source_dir: Path, headers_dir: Path | None = None,
                  extra_csv: Path | None = None) -> pd.DataFrame:
    d = read_feature_table(source_dir)
    y = d.groupby('Record', sort=True)['y'].agg(['first','nunique'])
    if not (y['nunique'] == 1).all():
        raise ValueError('Inconsistent outcome labels across channels')
    frame = y[['first']].rename(columns={'first':'y'})
    for family in BASE_FEATURES:
        pivot = d.pivot(index='Record', columns='Chann', values=family)
        for channel in (1,2,3):
            frame[f'{family}_{channel}'] = pivot[channel].astype(float)
    if extra_csv is not None and Path(extra_csv).exists():
        extra = pd.read_csv(extra_csv).set_index('Record')
        if extra.index.has_duplicates:
            raise ValueError('Duplicate records in signal feature cache')
        frame = frame.join(extra, how='left', validate='one_to_one')
    if headers_dir is not None and Path(headers_dir).is_dir():
        rows = []
        for record in frame.index:
            path = Path(headers_dir) / f'{record}.hea'
            if not path.exists():
                raise FileNotFoundError(path)
            h = read_header(path)
            rows.append({'Record':record, **{k:h.get(k) for k in CLINICAL+CATEGORICAL}})
        clinical = pd.DataFrame(rows).set_index('Record')
        frame = frame.join(clinical, validate='one_to_one')
    frame = frame.replace(['None','none','?', 'nan','NaN', 'inf', '-inf', ''], np.nan)
    for col in frame.columns:
        if col == 'y' or col in CATEGORICAL: continue
        frame[col] = pd.to_numeric(frame[col], errors='coerce').replace([np.inf,-np.inf],np.nan)
    frame = frame.reset_index()
    assert len(frame)==300 and frame['y'].sum()==38, 'Expected 262 term and 38 preterm records'
    assert frame['Record'].nunique()==len(frame)
    return frame.sort_values('Record').reset_index(drop=True)


def feature_columns(frame, kind='five', clinical=False):
    """Only predictors observable at the recording date, never delivery fields."""
    if kind=='five':
        cols = [f'{k}_{c}' for k in FEATURE_FAMILIES for c in (1,2,3) if f'{k}_{c}' in frame]
        # Frequency, RMS, zero-crossing and variance: paper's five selected families.
    elif kind=='all':
        cols = [c for c in frame.columns if c not in ('Record','y') and c not in CLINICAL+CATEGORICAL]
    else:
        raise ValueError(kind)
    if clinical:
        cols.extend(c for c in CLINICAL+CATEGORICAL if c in frame)
    if any(x in cols for x in ['Gestation', 'Group','Premature', 'Early','y','Record']):
        raise AssertionError('Outcome leakage into feature matrix')
    return cols

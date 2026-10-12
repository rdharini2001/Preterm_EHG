"""Features and model inputs extracted from TPEHGDB .dat/.hea WFDB recordings.

TPEHGDB files store 12 interleaved int16 channels at 20 Hz. Channels 3, 7, 11
(indexed 2, 6, 10) contain the supplied 0.3--3.0 Hz digitally filtered signals.
The WFDB ADC gain/baseline are read from the individual header files.
"""
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import welch, resample_poly
from .data import read_header

CHANNELS=(2,6,10)


def load_filtered_record(path: Path):
    head=Path(path).with_suffix('.hea')
    first=head.read_text().splitlines()
    parts=first[0].split()
    n_ch, fs, n=int(parts[1]),float(parts[2]),int(parts[3])
    if n_ch!=12 or abs(fs-20)>0.01:
        raise ValueError(f'Unexpected TPEHGDB format in {head}')
    raw=np.fromfile(Path(path).with_suffix('.dat'), dtype='<i2')
    if raw.size!=n_ch*n:
        raise ValueError(f'Malformed raw file: {path} ({raw.size} instead of {n_ch*n})')
    raw=raw.reshape(n,n_ch)
    channels=[]
    for c in CHANNELS:
        line=first[1+c].split()
        gain=float(line[2].split('/')[0]);baseline=float(line[4])
        channels.append((raw[:,c].astype(np.float32)-baseline)/gain)
    return np.stack(channels,axis=0),fs


def signal_features(raw_dir: Path, output_csv: Path, waveform_npz: Path | None=None,
                    window=750, shift=125, max_windows=40, signal_length=512,
                    trim_seconds: float=0.0):
    """Record-level statistics + compact time-series inputs with shared labels.

    Choose regularly spaced windows from all 30-minute recordings. Statistical
    estimates are aggregated *inside* each record prior to CV. Histogram counts
    are symbolic-Fourier descriptors, not an assertion of canonical BOSS parity.
    """
    rows=[]; snippets=[]; ids=[]
    for file in sorted(Path(raw_dir).glob('*.hea')):
        record=file.stem
        v,fs=load_filtered_record(file)
        if trim_seconds:
            cut=int(round(trim_seconds*fs))
            if 2*cut+window>v.shape[1]:
                raise ValueError(f'Trim too long for {record}')
            v=v[:,cut:-cut]
        starts=np.arange(0,v.shape[1]-window+1,shift)
        indices=np.unique(np.linspace(0,len(starts)-1,min(max_windows,len(starts)),dtype=int))
        chosen=starts[indices]
        out={'Record':record}
        for ch in range(3):
            blocks=np.stack([v[ch,s:s+window] for s in chosen]).astype(float)
            power=blocks**2
            # zero-crossing per second, with no artificial boundary transitions
            zc=((blocks[:,1:]*blocks[:,:-1])<0).sum(axis=1)/(window/fs)
            f,psd=welch(blocks,fs=fs,nperseg=window,axis=-1)
            mask=(f>=0.05)&(f<=3)
            fm=f[mask];p=psd[:,mask]
            med=fm[np.argmax(np.cumsum(p,axis=1)>=p.sum(axis=1,keepdims=True)/2,axis=1)]
            peak=fm[np.argmax(p,axis=1)]
            out.update({f'RMS_raw_{ch+1}':np.sqrt(power.mean(axis=1)).mean(),
                        f'zero_crossing_{ch+1}':zc.mean(),
                        f'variance_{ch+1}':blocks.var(axis=1).mean(),
                        f'Fmed_raw_{ch+1}':med.mean(),
                        f'Fpeak_raw_{ch+1}':peak.mean(),
                        f'spectral_entropy_{ch+1}':(-np.sum((p/np.maximum(p.sum(axis=1,keepdims=True),1e-12))*np.log(np.maximum(p/np.maximum(p.sum(axis=1,keepdims=True),1e-12),1e-12)),axis=1)).mean()})
            # Frequency-domain symbolic words, analogous to BOSS-like histograms
            # for 8 bins determined before fitting the classifier (no y involved).
            bins=np.clip(np.digitize(peak,[.1,.2,.3,.4,.6,.8,1.0]),0,7)
            counts=np.bincount(bins,minlength=8)/len(bins)
            for j,weight in enumerate(counts):out[f'symbolic_frequency_{ch+1}_{j}']=weight
        rows.append(out)
        # Five fixed segments per recording; resample 750 samples -> 512 for CNN/LSTM.
        if waveform_npz is not None:
            starts5=np.linspace(0,v.shape[1]-window,5,dtype=int)
            windows=np.stack([resample_poly(v[:,s:s+window],up=signal_length,down=window,axis=1) for s in starts5]).astype(np.float32)
            # z-normalize each segment with record-local statistics only
            windows=(windows-windows.mean(axis=-1,keepdims=True))/(windows.std(axis=-1,keepdims=True)+1e-6)
            snippets.append(windows);ids.append(record)
    if len(rows)!=300:raise ValueError(f'Expected 300 WFDB records, found {len(rows)}')
    Path(output_csv).parent.mkdir(parents=True,exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv,index=False,float_format='%.9g')
    if waveform_npz is not None:
        Path(waveform_npz).parent.mkdir(parents=True,exist_ok=True)
        np.savez_compressed(waveform_npz,record=np.array(ids),waves=np.stack(snippets))
    return pd.DataFrame(rows)

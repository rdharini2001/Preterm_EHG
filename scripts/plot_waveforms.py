"""Generate paired term/preterm EHG waveform plots from raw source recordings."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from tpehg.signals import load_filtered_record
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser();p.add_argument('--raw-dir',type=Path,default=Path('data/raw/tpehgdb'))
    p.add_argument('--out-dir',type=Path,default=Path('results/figures'))
    args=p.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True)
    for record,description in [('tpehg1007','Preterm'),('tpehg1021','Term')]:
        if not (args.raw_dir/f'{record}.dat').exists():continue
        v,fs=load_filtered_record(args.raw_dir/f'{record}.dat')
        seconds=100; n=min(int(seconds*fs),v.shape[1]);t=__import__('numpy').arange(n)/fs
        fig,ax=plt.subplots(3,1,figsize=(10,6),sharex=True,layout='constrained')
        for i,a in enumerate(ax):
            a.plot(t,v[i,:n],color=['#386b90','#67977e','#be7759'][i],lw=.7)
            a.set(ylabel=f'EHG ch. {i+1} (mV)');a.grid(alpha=.2)
        ax[0].set_title(f'{description} EHG recording — 0.3–3.0 Hz signal ({record})')
        ax[-1].set_xlabel('Time (seconds)')
        fig.savefig(args.out_dir/f'waveform_{description.lower()}.png',dpi=180)
        plt.close(fig)
if __name__=='__main__':main()

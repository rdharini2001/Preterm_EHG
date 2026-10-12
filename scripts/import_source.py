"""Import the full public TPEHGDB-Experiments source ZIP into this project.

Usage: python scripts/import_source.py /path/to/TPEHGDB-Experiments-master.zip
This script intentionally extracts only waveform .dat, .hea and .fvl files.
"""
from pathlib import Path
from zipfile import ZipFile
import argparse


def main():
    p=argparse.ArgumentParser()
    p.add_argument('archive',type=Path)
    p.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[1])
    a=p.parse_args()
    dest=a.root/'data/raw/tpehgdb'
    src=a.root/'data/source'
    dest.mkdir(parents=True,exist_ok=True);src.mkdir(parents=True,exist_ok=True)
    with ZipFile(a.archive) as z:
        count=0
        for entry in z.infolist():
            path=Path(entry.filename)
            if 'data' not in path.parts or path.suffix not in ('.hea','.dat','.fvl'):continue
            if path.parts[-2]=='tpehgdb' and path.suffix in ('.hea','.dat'):
                target=dest/path.name
            elif path.parts[-2]=='features' and path.suffix=='.fvl':
                target=src/path.name
            else: continue
            with z.open(entry) as fi,target.open('wb') as fo:
                import shutil
                shutil.copyfileobj(fi,fo)
            count+=1
    print(f'Imported {count} TPEHGDB source files to {a.root}')

if __name__=='__main__':main()

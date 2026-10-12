#!/usr/bin/env python3
"""Run an exploratory search over experimental protocols and evaluation rules."""
import argparse
from pathlib import Path
from tpehg.protocol_search import run
if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--root',type=Path,default=Path('.'))
    p.add_argument('--max-thresholds',type=int,default=101)
    a=p.parse_args()
    run(a.root.resolve(),a.max_thresholds)

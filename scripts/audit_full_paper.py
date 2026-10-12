"""Recalculate paper-number coverage from the saved model predictions and scores."""
from pathlib import Path
from tpehg.full_paper_audit import run
if __name__=='__main__':run(Path(__file__).resolve().parents[1])

from pathlib import Path
from tpehg.window_classifiers import run
if __name__=='__main__':
 print(run(Path(__file__).resolve().parents[1]))

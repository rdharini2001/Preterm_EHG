from pathlib import Path
from tpehg.stability import run
root=Path(__file__).resolve().parents[1]
print(run(root).to_string())

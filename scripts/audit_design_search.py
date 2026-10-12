from pathlib import Path
import json
from tpehg.design_coverage import run
print(json.dumps(run(Path(__file__).resolve().parents[1]),indent=2))

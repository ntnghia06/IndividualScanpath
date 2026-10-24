from pathlib import Path
import sys

BRANCH = Path(__file__).resolve().parents[1]
ROOT = next((p for p in BRANCH.parents
             if p.name == "PerGAZE" and (p / "GazeformerISP/src").is_dir()), None)
if ROOT is None or not (BRANCH.parent / "gazeformer_transfer").is_dir():
    raise RuntimeError("Keep Transfer/COCO-FV with its gazeformer_transfer package and base PerGAZE source")
BASE_SRC = ROOT / "GazeformerISP/src"
sys.path.insert(1, str(BRANCH.parent))
sys.path.insert(2, str(BASE_SRC))

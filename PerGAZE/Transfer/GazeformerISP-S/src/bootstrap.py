from pathlib import Path
import sys

BRANCH = Path(__file__).resolve().parents[1]
ROOT = BRANCH.parents[1]
BASE_SRC = ROOT / "GazeformerISP/src"
if not BASE_SRC.is_dir() or not (BRANCH.parent / "gazeformer_transfer").is_dir():
    raise RuntimeError("Keep this branch with Transfer/gazeformer_transfer and the full PerGAZE GazeformerISP source")
sys.path.insert(1, str(BRANCH.parent))
sys.path.insert(2, str(BASE_SRC))

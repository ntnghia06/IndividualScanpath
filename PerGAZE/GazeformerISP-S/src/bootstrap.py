from pathlib import Path
import sys

BASE_SRC = Path(__file__).resolve().parents[2] / "GazeformerISP/src"
if not BASE_SRC.is_dir():
    raise RuntimeError("GazeformerISP-S requires the sibling GazeformerISP source directory")
sys.path.insert(0, str(BASE_SRC))

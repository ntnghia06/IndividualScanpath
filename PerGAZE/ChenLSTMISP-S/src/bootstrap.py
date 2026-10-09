from pathlib import Path
import sys

BASE_SRC = Path(__file__).resolve().parents[2] / "ChenLSTMISP/src"
if not BASE_SRC.is_dir():
    raise RuntimeError("ChenLSTMISP-S requires the sibling ChenLSTMISP source directory")
sys.path.insert(0, str(BASE_SRC))

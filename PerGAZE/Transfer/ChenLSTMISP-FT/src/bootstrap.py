"""Import the existing PerGAZE architecture without modifying its source."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
BASE_SRC = ROOT / "ChenLSTMISP/src"
if not BASE_SRC.is_dir():
    raise RuntimeError("Keep Transfer/ChenLSTMISP-FT inside the full PerGAZE source tree")
sys.path.insert(1, str(BASE_SRC))

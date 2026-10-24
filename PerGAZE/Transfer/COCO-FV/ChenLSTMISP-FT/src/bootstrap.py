"""Import the PerGAZE architecture from this dataset-specific transfer branch."""
from pathlib import Path
import sys

ROOT = next((p for p in Path(__file__).resolve().parents
             if p.name == "PerGAZE" and (p / "ChenLSTMISP/src").is_dir()), None)
if ROOT is None:
    raise RuntimeError("Keep Transfer/COCO-FV with the full PerGAZE ChenLSTMISP source")
BASE_SRC = ROOT / "ChenLSTMISP/src"
sys.path.insert(1, str(BASE_SRC))

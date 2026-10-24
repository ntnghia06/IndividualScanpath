import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import bootstrap
from gazeformer_transfer.tests.test_transfer import TransferTests

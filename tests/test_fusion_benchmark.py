from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from evaluation.fusion_retrieval_benchmark import run_seed


def test_balanced_fusion_beats_item_mean_on_modality_imbalance_fixture() -> None:
    baseline = run_seed(7, "mean", groups=24, dimensions=32, image_duplicates=8)
    candidate = run_seed(7, "balanced", groups=24, dimensions=32, image_duplicates=8)

    assert candidate.recall_at_1 > baseline.recall_at_1
    assert candidate.mrr_at_5 > baseline.mrr_at_5

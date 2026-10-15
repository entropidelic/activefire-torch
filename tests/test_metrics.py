import math

import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, precision_score, recall_score

from activefire_torch.metrics import BINS, PixelCounts, ScoreHistogram


def test_counts_and_scores_by_hand():
    masks = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
    probabilities = torch.tensor([[0.9, 0.1, 0.3, 0.25]])
    # Predicted fire: 0.9 and 0.3 (0.25 is not strictly above the threshold).
    counts = PixelCounts()
    counts.add(probabilities, masks)
    assert counts.scores() == pytest.approx(
        {"precision": 0.5, "recall": 0.5, "f1": 0.5, "iou": 1 / 3, "tp": 1, "fp": 1, "fn": 1}
    )


def test_counts_are_summed_over_the_set_not_averaged_per_patch():
    large_fire_found = (torch.ones(1, 100), torch.ones(1, 100))  # 100 fire pixels, all found
    small_fire_missed = (torch.zeros(1, 1), torch.ones(1, 1))  # 1 fire pixel, missed
    counts = PixelCounts()
    for probabilities, masks in [large_fire_found, small_fire_missed]:
        counts.add(probabilities, masks)
    # Per-patch averaging would give recall (1 + 0) / 2 = 0.5.
    assert counts.scores()["recall"] == pytest.approx(100 / 101)


def test_predicting_no_fire_leaves_only_precision_undefined():
    counts = PixelCounts()
    counts.add(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    scores = counts.scores()
    assert math.isnan(scores["precision"])
    assert (scores["recall"], scores["f1"], scores["iou"]) == (0.0, 0.0, 0.0)


def test_no_fire_and_no_predictions_is_undefined():
    counts = PixelCounts()
    counts.add(torch.zeros(2, 1, 4, 4), torch.zeros(2, 1, 4, 4))
    scores = counts.scores()
    assert all(math.isnan(scores[name]) for name in ["precision", "recall", "f1", "iou"])
    assert (scores["tp"], scores["fp"], scores["fn"]) == (0, 0, 0)


def test_scores_match_sklearn():
    torch.manual_seed(0)
    masks = (torch.rand(3, 1, 16, 16) < 0.2).float()
    probabilities = torch.rand(3, 1, 16, 16)

    counts = PixelCounts()
    for i in range(3):  # one patch per batch
        counts.add(probabilities[i : i + 1], masks[i : i + 1])

    truth = masks.flatten().numpy()
    predicted = (probabilities > 0.25).flatten().numpy()
    scores = counts.scores()
    scores = {name: scores[name] for name in ["precision", "recall", "f1", "iou"]}
    assert scores == pytest.approx(
        {
            "precision": precision_score(truth, predicted),
            "recall": recall_score(truth, predicted),
            "f1": f1_score(truth, predicted),
            "iou": jaccard_score(truth, predicted),
        }
    )


def test_average_precision_by_hand():
    probabilities = torch.tensor([0.9, 0.8, 0.7, 0.1])
    masks = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Threshold below 0.9: 1 of 2 fires found, precision 1/1.
    # Threshold below 0.7: 2 of 2 fires found, precision 2/3.
    histogram = ScoreHistogram()
    histogram.add(probabilities, masks)
    assert histogram.average_precision() == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_perfect_scores_give_average_precision_1():
    masks = torch.tensor([1.0, 1.0, 0.0, 0.0])
    histogram = ScoreHistogram()
    histogram.add(masks * 0.8 + 0.1, masks)
    assert histogram.average_precision() == pytest.approx(1.0)


def test_average_precision_without_fire_is_undefined():
    histogram = ScoreHistogram()
    histogram.add(torch.rand(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert math.isnan(histogram.average_precision())


def random_predictions():
    """Masks with 10% fire, and noisy scores that are higher on fire pixels."""
    torch.manual_seed(0)
    masks = (torch.rand(4, 1, 64, 64) < 0.1).float()
    probabilities = (0.3 * masks + 0.7 * torch.rand(4, 1, 64, 64)).clamp(0, 1)
    return probabilities, masks


def test_average_precision_matches_sklearn_on_binned_scores():
    # With every score in the middle of a bin, binning loses nothing.
    probabilities, masks = random_predictions()
    probabilities = (torch.floor(probabilities * BINS) + 0.5) / BINS
    histogram = ScoreHistogram()
    for i in range(4):
        histogram.add(probabilities[i : i + 1], masks[i : i + 1])
    expected = average_precision_score(masks.flatten().numpy(), probabilities.flatten().numpy())
    assert histogram.average_precision() == pytest.approx(expected, abs=1e-9)


def test_average_precision_is_close_to_sklearn_on_raw_scores():
    probabilities, masks = random_predictions()
    histogram = ScoreHistogram()
    histogram.add(probabilities, masks)
    expected = average_precision_score(masks.flatten().numpy(), probabilities.flatten().numpy())
    assert histogram.average_precision() == pytest.approx(expected, abs=1e-3)

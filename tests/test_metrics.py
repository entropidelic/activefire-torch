import math

import pytest
import torch
from sklearn.metrics import average_precision_score, f1_score, jaccard_score, precision_score, recall_score

from activefire_torch.metrics import PixelCounts, ScoreHistogram


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


def test_undefined_scores_are_nan_not_zero():
    # Fire present, nothing predicted: only precision is undefined.
    counts = PixelCounts()
    counts.add(torch.zeros(1, 1, 4, 4), torch.ones(1, 1, 4, 4))
    scores = counts.scores()
    assert math.isnan(scores["precision"])
    assert (scores["recall"], scores["f1"], scores["iou"]) == (0.0, 0.0, 0.0)

    # No fire, nothing predicted: every score is undefined.
    counts = PixelCounts()
    counts.add(torch.zeros(1, 1, 4, 4), torch.zeros(1, 1, 4, 4))
    assert all(math.isnan(value) for name, value in counts.scores().items() if name not in ("tp", "fp", "fn"))


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
    logits = torch.tensor([2.0, 1.0, 0.5, -2.0])
    masks = torch.tensor([1.0, 0.0, 1.0, 0.0])
    # Threshold below 2.0: 1 of 2 fires found, precision 1/1.
    # Threshold below 0.5: 2 of 2 fires found, precision 2/3.
    histogram = ScoreHistogram()
    histogram.add(logits, masks)
    assert histogram.average_precision() == pytest.approx(0.5 * 1 + 0.5 * 2 / 3)


def test_average_precision_keeps_the_ranking_of_tiny_probabilities():
    # One fire pixel ranked above 9,999 background pixels, all with
    # probabilities below 0.001. Exact AP is 1; equal-width probability bins
    # would put them all in one bin and give 0.0001.
    logits = torch.cat([torch.tensor([-12.0]), torch.linspace(-20, -14, 9_999)])
    masks = torch.zeros(10_000)
    masks[0] = 1
    histogram = ScoreHistogram()
    histogram.add(logits, masks)
    assert histogram.average_precision() == pytest.approx(1.0)


def test_average_precision_matches_sklearn_on_realistic_scores():
    # Like a trained model: background far below 0, most fire above 0, and 5%
    # of the fire (missed fires) among the background. 0.5% of pixels are fire.
    torch.manual_seed(0)
    masks = (torch.rand(1_000_000) < 0.005).float()
    logits = torch.randn(1_000_000) * 3 - 12
    fire = masks == 1
    found = fire & (torch.rand(1_000_000) < 0.95)
    logits[found] = torch.randn(int(found.sum())) * 2 + 3
    logits[fire & ~found] = torch.randn(int((fire & ~found).sum())) * 2 - 9

    histogram = ScoreHistogram()
    histogram.add(logits, masks)
    expected = average_precision_score(masks.numpy(), logits.double().numpy())
    assert histogram.average_precision() == pytest.approx(expected, abs=1e-3)

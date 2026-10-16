"""Fire-class metrics over a whole evaluation set.

As in the original evaluate_v1.py: true positives, false positives and false
negatives are summed over every pixel of every patch, and each score is
computed once from the sums. Averaging per-patch scores instead would give a
patch with one fire pixel the same weight as a large fire.
"""

import torch

# TH_FIRE in the original inference.py. A pixel is fire if its probability is
# strictly above it, as there.
THRESHOLD = 0.25


class PixelCounts:
    """Running TP / FP / FN counts. Call add() once per batch, then scores()."""

    def __init__(self, threshold: float = THRESHOLD) -> None:
        self.threshold = threshold
        self.tp = 0
        self.fp = 0
        self.fn = 0

    def add(self, probabilities: torch.Tensor, masks: torch.Tensor) -> None:
        """probabilities and masks of the same shape, e.g. [N, 1, H, W]; masks in {0, 1}."""
        predicted = probabilities > self.threshold
        fire = masks == 1
        self.tp += int((predicted & fire).sum())
        self.fp += int((predicted & ~fire).sum())
        self.fn += int((~predicted & fire).sum())

    def scores(self) -> dict[str, float]:
        """Precision, recall, F1, IoU and the counts they come from.

        A score with a zero denominator is undefined and returned as nan, not 0:
        e.g. precision when nothing was predicted as fire. F1 is computed from
        the counts (equal to 2PR / (P + R) whenever that is defined), so it is
        only undefined, like IoU, when there is neither fire nor any prediction.
        """
        tp, fp, fn = self.tp, self.fp, self.fn
        return {
            "precision": divide(tp, tp + fp),
            "recall": divide(tp, tp + fn),
            "f1": divide(2 * tp, 2 * tp + fp + fn),
            "iou": divide(tp, tp + fp + fn),
            "tp": tp,
            "fp": fp,
            "fn": fn,
        }


def divide(numerator: int, denominator: int) -> float:
    """numerator / denominator, or nan if the denominator is 0."""
    return numerator / denominator if denominator > 0 else float("nan")


# Average precision needs precision and recall at every threshold. Sorting every
# pixel by score does not fit in memory (billions of pixels on the test set), so
# scores are counted into bins instead. The bins are over logits, not
# probabilities: almost all background pixels have probabilities below 0.001,
# which equal-width probability bins would lump into one bin (measured error
# -0.0085 against exact AP; with these logit bins, -0.00003). Logits outside
# [-LOGIT_LIMIT, LOGIT_LIMIT] go into the first or last bin.
BINS = 10_000
LOGIT_LIMIT = 40.0


class ScoreHistogram:
    """Running count of fire and background pixels per logit bin.
    Call add() once per batch, then average_precision()."""

    def __init__(self, bins: int = BINS) -> None:
        self.bins = bins
        self.fire = torch.zeros(bins, dtype=torch.int64)
        self.background = torch.zeros(bins, dtype=torch.int64)

    def add(self, logits: torch.Tensor, masks: torch.Tensor) -> None:
        """logits (the model's outputs, before the sigmoid) and masks of the same
        shape, e.g. [N, 1, H, W]; masks in {0, 1}."""
        position = (logits + LOGIT_LIMIT) / (2 * LOGIT_LIMIT)  # 0 at -LOGIT_LIMIT, 1 at +LOGIT_LIMIT
        bin_index = (position * self.bins).floor().long().clamp(0, self.bins - 1).flatten().cpu()
        fire = (masks == 1).flatten().cpu()
        self.fire += torch.bincount(bin_index[fire], minlength=self.bins)
        self.background += torch.bincount(bin_index[~fire], minlength=self.bins)

    def average_precision(self) -> float:
        """Area under the precision-recall curve, as sklearn's average_precision_score
        with logits rounded down to bin edges. nan if there is no fire.

        The threshold moves down one bin at a time. Each step adds that bin's
        pixels to the predicted fire: recall grows by the bin's share of all
        fire pixels, and that gain is weighted by the precision at that point.
        """
        fire = self.fire.tolist()
        background = self.background.tolist()
        total_fire = sum(fire)
        if total_fire == 0:
            return float("nan")

        average_precision = 0.0
        tp = 0
        fp = 0
        for b in reversed(range(self.bins)):
            tp += fire[b]
            fp += background[b]
            if fire[b] > 0:
                average_precision += (fire[b] / total_fire) * (tp / (tp + fp))
        return average_precision

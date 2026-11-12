"""Fire-class metrics over a whole evaluation set.

As in the original evaluate_v1.py: true positives, false positives and false
negatives are summed over every pixel of every patch, and each score is
computed once from the sums. Averaging per-patch scores instead would give a
patch with one fire pixel the same weight as a large fire.
"""

import math

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
        return scores_from_counts(self.tp, self.fp, self.fn)


def scores_from_counts(tp: int, fp: int, fn: int) -> dict[str, float]:
    """Precision, recall, F1 and IoU from pixel counts, with the counts (see PixelCounts.scores)."""
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
        # Clamped before the conversion to whole numbers, which is undefined for infinite logits.
        bin_index = (position * self.bins).floor().clamp(0, self.bins - 1).long().flatten().cpu()
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

    def best_threshold(self) -> float | None:
        """The probability threshold with the highest F1 on the pixels added so
        far; None if there is no fire. For choosing a threshold on validation
        data, to use on other data with scores_at().

        Thresholds are tried at the bin edges: "fire if the logit is in bin b
        or above". When a row of neighboring edges share the highest F1 (no
        pixel falls between them), the middle one is returned, the furthest
        from the pixels on either side.
        """
        total_fire = int(self.fire.sum())
        if total_fire == 0:
            return None
        # tp[b], fp[b]: fire and background pixels in bin b or above.
        tp = self.fire.flip(0).cumsum(0).flip(0)
        fp = self.background.flip(0).cumsum(0).flip(0)
        f1 = 2 * tp / (tp + fp + total_fire)  # 2 tp / (2 tp + fp + fn), with fn = total_fire - tp
        best_bins = (f1 == f1.max()).nonzero().flatten().tolist()
        row = [best_bins[0]]
        for b in best_bins[1:]:
            if b != row[-1] + 1:
                break
            row.append(b)
        best_bin = row[len(row) // 2]
        edge = -LOGIT_LIMIT + best_bin * (2 * LOGIT_LIMIT / self.bins)  # the lowest logit in best_bin
        return 1 / (1 + math.exp(-edge))  # sigmoid: the edge as a probability

    def scores_at(self, threshold: float) -> dict[str, float]:
        """Precision, recall, F1, IoU and counts (as PixelCounts.scores) with
        this probability threshold, counted from the bins.

        threshold  from best_threshold(). Any other value between 0 and 1 is
                   moved to the nearest bin edge, so the scores can differ
                   slightly from PixelCounts at that threshold.
        """
        if not 0 < threshold < 1:
            raise ValueError(f"threshold must be between 0 and 1, not {threshold}")
        logit = math.log(threshold / (1 - threshold))
        first_fire_bin = round((logit + LOGIT_LIMIT) / (2 * LOGIT_LIMIT) * self.bins)
        first_fire_bin = min(max(first_fire_bin, 0), self.bins)
        tp = int(self.fire[first_fire_bin:].sum())
        fp = int(self.background[first_fire_bin:].sum())
        fn = int(self.fire[:first_fire_bin].sum())
        return scores_from_counts(tp, fp, fn)

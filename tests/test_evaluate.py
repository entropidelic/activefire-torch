import numpy as np
import pandas as pd
import torch
from torch import nn

from activefire_torch.dataset import FireDataset, select_rows
from activefire_torch.evaluate import evaluate_algorithm, evaluate_model
from helpers import write_tif


def mask_with_fire_at(folder, name, pixels):
    mask = np.zeros((1, 4, 4), dtype=np.uint8)
    for row, column in pixels:
        mask[0, row, column] = 1
    write_tif(folder / name, mask)
    return str(folder / name)


def write_manual_split(folder):
    """Scene A: manual fire at (0, 0), image bright there; Schroeder finds (0, 0)
    and (0, 1). Scene B has no manual fire; Schroeder finds (1, 1) there."""
    rows = []
    for scene in ["A", "B"]:
        image = np.zeros((10, 4, 4), dtype=np.uint16)
        if scene == "A":
            image[:, 0, 0] = 60000
        write_tif(folder / f"{scene}.tif", image)
        rows.append({"image": str(folder / f"{scene}.tif"), "scene_id": scene, "split": "test"})
    rows[0]["mask_manual"] = mask_with_fire_at(folder, "A_manual.tif", [(0, 0)])
    rows[0]["mask_schroeder"] = mask_with_fire_at(folder, "A_schroeder.tif", [(0, 0), (0, 1)])
    rows[1]["mask_manual"] = None
    rows[1]["mask_schroeder"] = mask_with_fire_at(folder, "B_schroeder.tif", [(1, 1)])
    split_file = folder / "manual.csv"
    pd.DataFrame(rows).to_csv(split_file, index=False)
    return split_file


def test_evaluate_algorithm(tmp_path):
    rows = select_rows(write_manual_split(tmp_path), "test", "manual", "full")
    results = evaluate_algorithm(rows, "schroeder")
    assert (results["tp"], results["fp"], results["fn"]) == (1, 2, 0)
    assert results["fire_free_scenes_fp"] == 1  # the detection in scene B


class BrightIsFire(nn.Module):
    """Predicts fire wherever the first band is bright."""

    def forward(self, image):
        return (image[:, :1] - 0.5) * 100


def test_evaluate_model(tmp_path):
    dataset = FireDataset(write_manual_split(tmp_path), "test", ["B7"], "manual", "full")
    results = evaluate_model(BrightIsFire(), dataset, torch.device("cpu"), num_workers=0)
    assert (results["tp"], results["fp"], results["fn"]) == (1, 0, 0)
    assert results["fire_free_scenes_fp"] == 0
    assert results["average_precision"] == 1.0

#!/usr/bin/env python
"""utils.py

Shared helpers for mixed-frequency MED experiments
=================================================
* dataset builders (`create_dataset_two_freq`, `create_dataset_three_freq`)
* metrics (`sMAPE`)
* reproducibility (`fix_random_seed`)
* PyTorch `CustomDatasetD`
* basic plotting utilities (predictions, attention heat-maps)

Author : Jiaxi Liu
License: MIT
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset


# -----------------------------------------------------------------------------
# DATASET BUILDERS
# -----------------------------------------------------------------------------

def create_dataset_two_freq(
        data: Tuple[np.ndarray, np.ndarray],
        *,
        freq_r: int = 3,
        input_steps: int = 1,
        t_output_steps: int = 1,
        a_output_steps: int = 1,
        target_idx: int = 0,
):
    """Build (X_t, X_a, y_t, y_a) tensors from two numpy arrays.

    Parameters
    ----------
    data : ([aux_array], [target_array]) – *already scaled* numpy matrices.
    freq_r : int  Frequency ratio (target / aux). 3 ≅ monthly ↔ quarterly.
    input_steps, t_output_steps, a_output_steps : window lengths.
    target_idx : which column of target array to predict.
    """
    X_t, X_a, y_t, y_a = [], [], [], []
    i = 0
    while i < len(data[1]) - input_steps:
        X_t.append(data[1][i: i + input_steps])
        X_a.append(data[0][int(i // freq_r): int(i // freq_r) + int(input_steps // freq_r), :])
        y_t.append(data[1][i + input_steps: i + input_steps + t_output_steps, target_idx])
        y_a.append(data[0][int(i // freq_r) + int(input_steps // freq_r)
                           : int(i // freq_r) + int(input_steps // freq_r) + a_output_steps, :])
        i += 1
    return map(np.asarray, (X_t, X_a, y_t, y_a))


def create_dataset_three_freq(
        data: Tuple[np.ndarray, np.ndarray, np.ndarray],
        *,
        freq_r1: int = 3,
        freq_r2: int = 6,
        input_steps: int = 1,
        t_output_steps: int = 1,
        a1_output_steps: int = 1,
        a2_output_steps: int = 1,
        target_idx: int = 0,
):
    """Three-frequency variant (aux1, aux2, target)."""
    X_t, X_a1, X_a2, y_t, y_a1, y_a2 = [], [], [], [], [], []
    i = 0
    while i < len(data[2]) - input_steps:
        X_t.append(data[2][i: i + input_steps])
        X_a1.append(data[0][int(i // freq_r1): int(i // freq_r1) + input_steps // freq_r1])
        X_a2.append(data[1][int(i // freq_r2): int(i // freq_r2) + input_steps // freq_r2])

        y_t.append(data[2][i + input_steps: i + input_steps + t_output_steps, target_idx])
        y_a1.append(
            data[0][
            int(i // freq_r1)
            + input_steps // freq_r1: int(i // freq_r1)
                                      + input_steps // freq_r1
                                      + a1_output_steps
            ]
        )
        y_a2.append(
            data[1][
            int(i // freq_r2)
            + input_steps // freq_r2: int(i // freq_r2)
                                      + input_steps // freq_r2
                                      + a2_output_steps
            ]
        )
        i += 1
    return map(np.asarray, (X_t, X_a1, X_a2, y_t, y_a1, y_a2))


# -----------------------------------------------------------------------------
# METRICS
# -----------------------------------------------------------------------------

def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    if y_true.ndim > 1:
        y_true = y_true.ravel()
    if y_pred.ndim > 1:
        y_pred = y_pred.ravel()
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    diff = np.abs(y_true - y_pred) / denom
    diff[denom == 0] = 0.0
    return np.mean(diff)


# -----------------------------------------------------------------------------
# REPRODUCIBILITY
# -----------------------------------------------------------------------------

def fix_random_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# -----------------------------------------------------------------------------
# PYTORCH DATASET
# -----------------------------------------------------------------------------

class CustomDatasetD(Dataset):
    """Simple tensor-dict dataset for dual-frequency sequences."""

    def __init__(
            self,
            X_enc_t,
            X_enc_a,
            X_dec_t,
            X_dec_a,
            y_t,
            y_a,
    ) -> None:
        self.X_enc_t = torch.as_tensor(X_enc_t, dtype=torch.float32)
        self.X_enc_a = torch.as_tensor(X_enc_a, dtype=torch.float32)
        self.X_dec_t = torch.as_tensor(X_dec_t, dtype=torch.float32)
        self.X_dec_a = torch.as_tensor(X_dec_a, dtype=torch.float32)
        self.y_t = torch.as_tensor(y_t, dtype=torch.float32)
        self.y_a = torch.as_tensor(y_a, dtype=torch.float32)

    def __len__(self):
        return len(self.X_enc_t)

    def __getitem__(self, idx):  # noqa: D401
        return {
            "X_enc_t": self.X_enc_t[idx],
            "X_enc_a": self.X_enc_a[idx],
            "X_dec_t": self.X_dec_t[idx],
            "X_dec_a": self.X_dec_a[idx],
            "y_train_t": self.y_t[idx],
            "y_train_a": self.y_a[idx],
        }


# -----------------------------------------------------------------------------
# PLOTTING
# -----------------------------------------------------------------------------

def plot_predictions_and_real_values(preds, reals, *, title: str = "pred_vs_real", save_dir: str = "figures"):
    Path(save_dir).mkdir(exist_ok=True)
    plt.figure(figsize=(12, 6))
    plt.plot(reals, "--", label="Real")
    plt.plot(preds, label="Pred")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(Path(save_dir) / f"{title}.png", dpi=300)
    plt.close()

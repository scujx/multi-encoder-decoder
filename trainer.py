#!/usr/bin/env python
"""model_trainer.py

Top-level orchestration script that wires data, the MED model wrapper and
training / evaluation loops together.

It supports
* K-fold training (delegated to :class:`med_model.MEDModel`)
* test-time evaluation and plotting
* metric logging + Excel dump via :pydata:`pandas`

Author : Jiaxi Liu
License: MIT
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, mean_squared_error

from models.med_model import MEDModel
from utils import (
    fix_random_seed,
    plot_predictions_and_real_values,
    symmetric_mean_absolute_percentage_error,
)

# -----------------------------------------------------------------------------

MetricDict = Dict[str, float]


class ModelTrainer:
    """End-to-end trainer for MED models (dual-frequency version)."""

    # ------------------------------------------------------------------
    # constructor
    # ------------------------------------------------------------------
    def __init__(
        self,
        train_data: Tuple[pd.DataFrame, pd.DataFrame],
        test_data: Tuple[pd.DataFrame, pd.DataFrame],
        args,
    ) -> None:
        fix_random_seed(args.seed)
        self.args = args
        self.train_data = train_data
        self.test_data = test_data

        # -------------------------------------------------------------- #
        self.model = MEDModel(args)
        n_param = sum(p.numel() for p in self.model.model.parameters() if p.requires_grad)
        print(f"Model params: {n_param / 1e6:.2f} M")

        self.key_var = args.key_var
        self.metrics: Dict[str, MetricDict] = {}
        self.real_series: Dict[str, str] = {}
        self.pred_series: Dict[str, str] = {}

    # ------------------------------------------------------------------
    #  TRAIN + TEST
    # ------------------------------------------------------------------
    def train(self) -> None:
        """K-fold training via :class:`MEDModel`."""
        self.model.train(self.train_data)

    def test(self) -> None:
        """Generate predictions and compute regression metrics."""
        preds = self.model.test(self.test_data, self.train_data)
        reals = self.test_data[1][self.key_var].values  # quarterly target

        mae = mean_absolute_error(reals, preds)
        mape = mean_absolute_percentage_error(reals, preds)
        mse = mean_squared_error(reals, preds)
        smape = symmetric_mean_absolute_percentage_error(reals, preds)

        self.metrics[self.key_var] = {"mae": mae, "mape": mape, "mse": mse, "smape": smape}
        self.real_series[self.key_var] = ",".join(map(str, reals))
        self.pred_series[self.key_var] = ",".join(map(str, preds))

        if self.args.plot:
            plot_predictions_and_real_values(preds, reals, title=self.args.model_type)

        print(f"MAE={mae:.4f} | MAPE={mape:.4f} | MSE={mse:.4f} | sMAPE={smape:.4f}")

"""med_model.py

High-level training / evaluation wrapper for the dual-frequency MED network.

It handles:
* dataset scaling + creation (`create_dataset_two_freq`, `CustomDatasetD`)
* K-Fold training with early stopping
* test-time stitching (uses last encoder window from train set)

Only the **two-frequency** case is exposed; extending to more streams merely
requires minor plumbing.  Designed for clarity in the public GitHub release.

Author : Jiaxi Liu <liujiaxi@stu.scu.edu.cn>
License: MIT
"""
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from torch import Tensor, nn
from torch.utils.data import DataLoader

from models.dual_encoder_decoder import MED  # local import
from utils import (
    CustomDatasetD,
    create_dataset_two_freq,
    fix_random_seed,
)


class MEDModel:  # pylint: disable=too-many-instance-attributes
    """Wrapper around :class:`models.dual_encoder_decoder.MED` with utilities."""

    # ---------------------------------------------------------------------
    # constructor
    # ---------------------------------------------------------------------
    def __init__(self, args) -> None:  # args comes from CLI/dataclass
        fix_random_seed(args.seed)
        self.args = args
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------- hyper-params from args ---------------------------------- #
        self.epoch = args.epoch
        self.batch_size = args.batch_size
        self.loss_alpha = args.loss_alpha

        # encoder / decoder window sizes
        self.t_enc_steps = args.t_enc_time_steps
        self.a_enc_steps = args.a_enc_time_steps
        self.t_dec_steps = args.t_dec_time_steps
        self.a_dec_steps = args.a_dec_time_steps
        self.t_out_steps = args.t_output_steps
        self.a_out_steps = args.a_output_steps
        self.freq_ratio = args.t_a1_freq_ratio

        # scalers ---------------------------------------------------------- #
        self.scaler_t = MinMaxScaler()
        self.scaler_a = MinMaxScaler()

        # network & optimisation ------------------------------------------ #
        self.model = MED(args).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.learning_rate)
        self.criterion = nn.MSELoss()
        self.grad_clip = 1.0

        # bookkeeping
        self.key_var = args.key_var  # target dimension name
        self.key_idx: int = 0  # will be resolved after reading data

    # ------------------------------------------------------------------
    # TRAINING (K-fold)
    # ------------------------------------------------------------------
    def train(self, train_data: Tuple[pd.DataFrame, pd.DataFrame]) -> None:
        """Fit the network with 5-fold cross-validation + early stopping."""

        df_a, df_t = train_data
        self.key_idx = df_t.columns.get_loc(self.key_var)

        # scale
        df_a = self.scaler_a.fit_transform(df_a)
        df_t = self.scaler_t.fit_transform(df_t)

        # build tensors (N, win, F)
        X_t, X_a, y_t, y_a = create_dataset_two_freq(
            [df_a, df_t],
            freq_r=self.freq_ratio,
            input_steps=self.t_enc_steps,
            t_output_steps=self.t_out_steps,
            a_output_steps=self.a_out_steps,
            target_idx=self.key_idx,
        )

        kfold = KFold(n_splits=5, shuffle=False)
        patience, best_val = 10, float("inf")

        for fold, (tr_idx, val_idx) in enumerate(kfold.split(X_t), 1):
            train_loader = DataLoader(
                CustomDatasetD(
                    X_t[tr_idx],
                    X_a[tr_idx],
                    X_t[tr_idx, -self.t_dec_steps:],
                    X_a[tr_idx, -self.a_dec_steps:],
                    y_t[tr_idx],
                    y_a[tr_idx],
                ),
                batch_size=self.batch_size,
                shuffle=True,
            )
            val_loader = DataLoader(
                CustomDatasetD(
                    X_t[val_idx],
                    X_a[val_idx],
                    X_t[val_idx, -self.t_dec_steps:],
                    X_a[val_idx, -self.a_dec_steps:],
                    y_t[val_idx],
                    y_a[val_idx],
                ),
                batch_size=self.batch_size,
                shuffle=False,
            )

            patience_ctr = 0
            for epoch in range(1, self.epoch + 1):
                # ---- train one epoch ---- #
                self.model.train()
                tr_aux, tr_tgt = self._run_epoch(train_loader, train=True)

                # ---- validation --------- #
                val_tgt, val_aux, *_ = self._run_epoch(val_loader, train=False)
                val_total = val_tgt + self.loss_alpha * val_aux

                if self.args.verbose and epoch % 10 == 0:
                    print(
                        f"Fold {fold} | Ep {epoch:03d} | "
                        f"train={tr_tgt + tr_aux:.4f} | val={val_total:.4f}"
                    )

                # early stopping
                if val_total < best_val:
                    best_val = val_total
                    patience_ctr = 0
                else:
                    patience_ctr += 1
                if patience_ctr >= patience:
                    if self.args.verbose:
                        print(f"Fold {fold}: early stop @ epoch {epoch}")
                    break

    # ------------------------------------------------------------------
    # TEST / INFERENCE
    # ------------------------------------------------------------------
    def test(
            self,
            test_data: Tuple[pd.DataFrame, pd.DataFrame],
            train_tail: Tuple[pd.DataFrame, pd.DataFrame],
    ) -> np.ndarray:
        """Return inverse-scaled predictions of the key variable."""

        df_a = pd.concat([train_tail[0].tail(self.a_enc_steps), test_data[0]])
        df_t = pd.concat([train_tail[1].tail(self.t_enc_steps), test_data[1]])

        df_a = self.scaler_a.transform(df_a)
        df_t = self.scaler_t.transform(df_t)

        X_t, X_a, y_t, y_a = create_dataset_two_freq(
            [df_a, df_t],
            freq_r=self.freq_ratio,
            input_steps=self.t_enc_steps,
            t_output_steps=self.t_out_steps,
            a_output_steps=self.a_out_steps,
            target_idx=self.key_idx,
        )

        loader = DataLoader(
            CustomDatasetD(
                X_t,
                X_a,
                X_t[:, -self.t_dec_steps:],
                X_a[:, -self.a_dec_steps:],
                y_t,
                y_a,
            ),
            batch_size=self.batch_size,
            shuffle=False,
        )

        pred_key = self._predict(loader)
        # inverse scale
        dummy = np.zeros((len(pred_key), df_t.shape[1]))
        dummy[:, self.key_idx] = pred_key
        return self.scaler_t.inverse_transform(dummy)[:, self.key_idx]

    # ------------------------------------------------------------------
    # INTERNAL HELPERS
    # ------------------------------------------------------------------
    def _run_epoch(self, loader: DataLoader, *, train: bool) -> Tuple[float, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()
        aux_loss, tgt_loss = 0.0, 0.0

        with torch.set_grad_enabled(train):
            for batch in loader:
                t_src = batch["X_enc_t"].to(self.device)
                a_src = batch["X_enc_a"].to(self.device)
                t_trg = batch["X_dec_t"].to(self.device)
                a_trg = batch["X_dec_a"].to(self.device)
                t_y = batch["y_train_t"].to(self.device)
                a_y = batch["y_train_a"].to(self.device)

                t_out, a_out = self.model(t_src, a_src, t_trg, a_trg)
                t_out = t_out[:, -1]
                a_out = a_out[:, -1]

                loss_t = self.criterion(t_out, t_y.view_as(t_out))
                loss_a = self.criterion(a_out, a_y.view_as(a_out))

                if train:
                    loss = loss_t + self.loss_alpha * loss_a
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
                    self.optimizer.step()
                    self.optimizer.zero_grad()

                aux_loss += loss_a.item()
                tgt_loss += loss_t.item()

        n_batches = len(loader)
        return tgt_loss / n_batches, aux_loss / n_batches

    def _predict(self, loader: DataLoader) -> np.ndarray:
        self.model.eval()
        preds: List[Tensor] = []
        with torch.no_grad():
            for batch in loader:
                t_src = batch["X_enc_t"].to(self.device)
                a_src = batch["X_enc_a"].to(self.device)
                t_trg = batch["X_dec_t"].to(self.device)
                a_trg = batch["X_dec_a"].to(self.device)
                t_out, _ = self.model(t_src, a_src, t_trg, a_trg)
                preds.append(t_out[:, -1].cpu())
        return torch.cat(preds).squeeze().numpy()

# -----------------------------------------------------------------------------
# NOTE: File name changed from "MED.py" to "med_model.py" to avoid uppercase
# clashes on case-insensitive file systems and to match PEP-8 module naming.
# -----------------------------------------------------------------------------

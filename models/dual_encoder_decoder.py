"""dual_encoder_decoder.py

Minimal public implementation of the *Dual-frequency Multi-Encoder–Decoder* (MED)
model. Two separate LSTM encoders/decoders operate on target-frequency and
auxiliary-frequency series, respectively; MFA combines the latent states.

Only the dual-frequency variant is released here to keep the repository concise;
three-frequency extensions follow exactly the same design pattern and can be
added later.

Author: Jiaxi Liu  <liujiaxi@stu.scu.edu.cn>
License: MIT
"""
from __future__ import annotations

from typing import Tuple, List

import torch
from torch import nn, Tensor

from models.rnn_components import (
    Attention,
    LSTMDecoder,
    LSTMEncoder,
    Seq2Seq,
)
import utils  # project-level utility module (seed fixer, etc.)

# -----------------------------------------------------------------------------
# 1.  TOP-LEVEL MED MODULE
# -----------------------------------------------------------------------------


class MED(nn.Module):
    """Dual-frequency MED wrapper with training / evaluation helpers."""

    def __init__(self, args) -> None:  # args originates from argparse or OmegaConf
        super().__init__()
        utils.fix_random_seed(args.seed)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # -------------------- build encoders -------------------- #
        t_encoder = LSTMEncoder(
            input_dim=args.t_d_model,
            hidden_dim=args.t_enc_hd,
            num_layers=args.t_enc_num_layers,
            dropout=args.t_enc_dropout,
        ).to(self.device)

        a_encoder = LSTMEncoder(
            input_dim=args.a1_d_model,
            hidden_dim=args.a_enc_hd,
            num_layers=args.a_enc_num_layers,
            dropout=args.a_enc_dropout,
        ).to(self.device)

        # -------------------- build decoders -------------------- #
        t_decoder = LSTMDecoder(
            output_dim=1,  # univariate target
            input_dim=args.t_d_model,
            enc_hidden_dim=args.t_enc_hd,
            dec_hidden_dim=args.t_dec_hd,
            num_layers=args.t_dec_num_layers,
            dropout=args.t_dec_dropout,
            attention=Attention(args.t_enc_hd, args.t_dec_hd),
        ).to(self.device)

        a_decoder = LSTMDecoder(
            output_dim=args.a1_d_model,  # predict next auxiliary frame as aux loss
            input_dim=args.a1_d_model,
            enc_hidden_dim=args.a_enc_hd,
            dec_hidden_dim=args.a_dec_hd,
            num_layers=args.a_dec_num_layers,
            dropout=args.a_dec_dropout,
            attention=Attention(args.a_enc_hd, args.a_dec_hd),
        ).to(self.device)

        # -------------------- Seq2Seq container ----------------- #
        self.seq2seq = Seq2Seq(
            t_encoder,
            a_encoder,
            t_decoder,
            a_decoder,
            t_enc_hidden_dim=args.t_enc_hd,
            a_enc_hidden_dim=args.a_enc_hd,
            t_dec_hidden_dim=args.t_dec_hd,
            a_dec_hidden_dim=args.a_dec_hd,
            device=self.device,
            seed=args.seed,
            alpha=args.loss_alpha,
            transform_type=args.transform_type,
            mfa_in_dim=args.a_enc_time_steps,
            mfa_d_model=args.t_enc_time_steps,
            mfa_heads=args.mfa_heads
        ).to(self.device)

        self.loss_alpha = args.loss_alpha

    # ---------------------------------------------------------------------
    # training loop for a single epoch
    # ---------------------------------------------------------------------
    def train_epoch(
        self,
        loader,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        clip: float = 1.0,
    ) -> Tuple[float, float]:
        self.seq2seq.train()
        loss_aux, loss_tgt = 0.0, 0.0

        for batch in loader:
            t_src: Tensor = batch["X_enc_t"].to(self.device)
            a_src: Tensor = batch["X_enc_a"].to(self.device)
            t_trg: Tensor = batch["X_dec_t"].to(self.device)
            a_trg: Tensor = batch["X_dec_a"].to(self.device)
            t_y: Tensor = batch["y_train_t"].to(self.device)
            a_y: Tensor = batch["y_train_a"].to(self.device)

            optimizer.zero_grad()
            t_out, a_out = self.seq2seq(t_src, a_src, t_trg, a_trg)

            # use last-step prediction for loss
            t_out = t_out[:, -1]
            a_out = a_out[:, -1]

            loss_t = criterion(t_out, t_y.view_as(t_out))
            loss_a = criterion(a_out, a_y.view_as(a_out))
            loss = loss_t + self.loss_alpha * loss_a
            loss.backward()

            nn.utils.clip_grad_norm_(self.parameters(), clip)
            optimizer.step()

            loss_aux += loss_a.item()
            loss_tgt += loss_t.item()

        n_batches = len(loader)
        return loss_aux / n_batches, loss_tgt / n_batches

    # ---------------------------------------------------------------------
    # evaluation (no teacher forcing)
    # ---------------------------------------------------------------------
    def evaluate(
        self,
        loader,
        criterion: nn.Module,
    ) -> Tuple[float, float, List[Tensor], List[Tensor]]:
        self.seq2seq.eval()
        loss_tgt, loss_aux = 0.0, 0.0
        preds, reals = [], []

        with torch.no_grad():
            for batch in loader:
                t_src = batch["X_enc_t"].to(self.device)
                a_src = batch["X_enc_a"].to(self.device)
                t_trg = batch["X_dec_t"].to(self.device)
                a_trg = batch["X_dec_a"].to(self.device)
                t_y = batch["y_train_t"].to(self.device)
                a_y = batch["y_train_a"].to(self.device)

                t_out, a_out = self.seq2seq(t_src, a_src, t_trg, a_trg)  # teacher forcing = 0 inside
                t_out = t_out[:, -1]
                a_out = a_out[:, -1]

                loss_t = criterion(t_out, t_y.view_as(t_out))
                loss_a = criterion(a_out, a_y.view_as(a_out))
                loss_tgt += loss_t.item()
                loss_aux += loss_a.item()

                preds.append(t_out.cpu())
                reals.append(t_y.view_as(t_out).cpu())

        n_batches = len(loader)
        return loss_tgt / n_batches, loss_aux / n_batches, preds, reals

    def forward(self, t_src, a_src, t_trg, a_trg):
        """Thin wrapper that calls the underlying Seq2Seq network."""
        return self.seq2seq(t_src, a_src, t_trg, a_trg)
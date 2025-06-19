#!/usr/bin/env python
"""rnn_components.py

Reusable RNN / attention layers for mixed-frequency Seq2Seq models.

 * **LSTMEncoder / LSTMDecoder** with optional Bahdanau-style attention
 * **MFALayer**: a Transformer-encoder layer that stores its attention
   weights for later visualisation ("MFA" = Multi-Frequency Attention)
 * **Seq2Seq** wrapper that supports two input streams (target frequency
   + auxiliary frequency) and optional MFA fusion.

Author: Jiaxi Liu
License: MIT
"""
from __future__ import annotations

import random
from typing import Tuple, Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn


# -----------------------------------------------------------------------------
# 1.  LSTM ENCODER
# -----------------------------------------------------------------------------


class LSTMEncoder(nn.Module):
    """Standard *batch_first* LSTM encoder."""

    def __init__(
            self,
            input_dim: int,
            hidden_dim: int,
            num_layers: int = 1,
            dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.rnn = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:  # type: ignore[name-defined]
        # x : (B, T, input_dim)
        outputs, (hidden, cell) = self.rnn(x)  # outputs (B, T, hidden_dim)
        hidden = self.dropout(hidden)
        return outputs, (hidden, cell)


# -----------------------------------------------------------------------------
# 2.  BAHADANAU ATTENTION (additive)
# -----------------------------------------------------------------------------

class Attention(nn.Module):
    """Additive attention with *batch_first* inputs."""

    def __init__(self, enc_hidden_dim: int, dec_hidden_dim: int):
        super().__init__()
        self.attn = nn.Linear(enc_hidden_dim + dec_hidden_dim, dec_hidden_dim)
        self.v = nn.Linear(dec_hidden_dim, 1, bias=False)

    def forward(self, hidden: Tensor, encoder_outputs: Tensor) -> Tensor:  # type: ignore[name-defined]
        # hidden: (num_layers, B, dec_hidden_dim) → take last layer
        # encoder_outputs: (B, src_len, enc_hidden_dim)
        src_len = encoder_outputs.size(1)
        hidden = hidden[-1].unsqueeze(1).repeat(1, src_len, 1)  # (B, src_len, dec_h)
        energy = torch.tanh(self.attn(torch.cat((hidden, encoder_outputs), dim=2)))
        scores = self.v(energy).squeeze(2)  # (B, src_len)
        return F.softmax(scores, dim=1)


# -----------------------------------------------------------------------------
# 3.  LSTM DECODER WITH ATTENTION
# -----------------------------------------------------------------------------

class LSTMDecoder(nn.Module):
    """LSTM decoder that attends over encoder outputs at every step."""

    def __init__(
            self,
            output_dim: int,
            input_dim: int,
            enc_hidden_dim: int,
            dec_hidden_dim: int,
            num_layers: int = 1,
            dropout: float = 0.0,
            attention: Optional[Attention] = None,
    ) -> None:
        super().__init__()
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.attention = attention or Attention(enc_hidden_dim, dec_hidden_dim)

        self.rnn = nn.LSTM(
            input_dim + enc_hidden_dim,
            dec_hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(dec_hidden_dim, dec_hidden_dim)
        self.fc_out = nn.Linear(enc_hidden_dim + dec_hidden_dim + input_dim, output_dim)

    def forward(
            self,
            x_t: Tensor,
            hidden: Tensor,
            encoder_outputs: Tensor,
            cell: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:  # type: ignore[name-defined]
        # x_t: (B, input_dim)
        x_t = x_t.unsqueeze(1)  # (B, 1, input_dim)
        attn_w = self.attention(hidden, encoder_outputs).unsqueeze(1)  # (B, 1, src_len)
        weighted = torch.bmm(attn_w, encoder_outputs)  # (B, 1, enc_hidden)

        # align hidden & cell dims for multi-layer RNN
        hidden = hidden[-1].unsqueeze(0).repeat(self.num_layers, 1, 1)
        cell = (cell if cell is not None else hidden)[-1].unsqueeze(0).repeat(
            self.num_layers, 1, 1
        )

        rnn_input = torch.cat((x_t, weighted), dim=2)
        output, (hidden, cell) = self.rnn(rnn_input, (hidden, cell))

        out = self.dropout(self.fc1(output[:, -1, :]))
        pred = self.fc_out(torch.cat((out, weighted.squeeze(1), x_t.squeeze(1)), dim=1))
        return pred, hidden, cell


# -----------------------------------------------------------------------------
# 4.  TRANSFORMER LAYER THAT SAVES ATTENTION WEIGHTS (FOR MFA)
# -----------------------------------------------------------------------------

class MFALayer(nn.TransformerEncoderLayer):
    """A vanilla TransformerEncoderLayer that stores its self-attention weights."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.saved_attn_weights: Optional[Tensor] = None

    # PyTorch ≥2 uses *is_causal*; keep kwargs for compatibility
    def forward(
            self,
            src: Tensor,
            src_mask: Optional[Tensor] = None,
            src_key_padding_mask: Optional[Tensor] = None,
            is_causal: bool = False,
            **kwargs,
    ) -> Tensor:
        attn_out, attn_w = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.saved_attn_weights = attn_w.detach().cpu()

        src = src + self.dropout1(attn_out)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = self.norm2(src + self.dropout2(src2))
        return src


# -----------------------------------------------------------------------------
# 5.  SEQ2SEQ WITH OPTIONAL MFA FUSION
# -----------------------------------------------------------------------------

class Seq2Seq(nn.Module):
    """Two-stream Seq2Seq model (target + auxiliary) with optional MFA."""

    def __init__(
            self,
            t_encoder: LSTMEncoder,
            a_encoder: LSTMEncoder,
            t_decoder: LSTMDecoder,
            a_decoder: LSTMDecoder,
            *,
            t_enc_hidden_dim: int,
            a_enc_hidden_dim: int,
            t_dec_hidden_dim: int,
            a_dec_hidden_dim: int,
            device: torch.device,
            seed: int | None = None,
            alpha: float = 0.9,
            transform_type: str = "mfa",
            mfa_in_dim: int = 4,
            mfa_d_model: int = 12,
            mfa_heads: int = 4,
    ) -> None:
        super().__init__()
        self.t_encoder, self.a_encoder = t_encoder, a_encoder
        self.t_decoder, self.a_decoder = t_decoder, a_decoder
        self.device = device
        self.alpha = alpha
        self.transform_type = transform_type.lower()

        # simple linear projections
        self.a_projection = nn.Linear(a_enc_hidden_dim, a_dec_hidden_dim)
        self.t_projection = nn.Linear(a_enc_hidden_dim + t_enc_hidden_dim, t_dec_hidden_dim)

        # Optional MFA block
        if self.transform_type == "mfa":
            self.mfa_projection = nn.Linear(mfa_in_dim, mfa_d_model)
            self.mfa_layer = MFALayer(
                d_model=mfa_d_model,
                nhead=mfa_heads,
                dim_feedforward=mfa_d_model * 2,
                batch_first=True,
            )
            self.mfa = nn.TransformerEncoder(self.mfa_layer, num_layers=1)
            self.h_proj = nn.Linear(mfa_d_model, t_dec_hidden_dim)
            self.c_proj = nn.Linear(mfa_d_model, t_dec_hidden_dim)

        if seed is not None:
            random.seed(seed)
            torch.manual_seed(seed)

    # ------------------------------------------------------------------
    # forward pass
    # ------------------------------------------------------------------
    def forward(
            self,
            t_src: Tensor,
            a_src: Tensor,
            t_trg: Tensor,
            a_trg: Tensor,
    ) -> Tuple[Tensor, Tensor]:  # type: ignore[name-defined]
        batch_size, t_len, _ = t_trg.shape
        _, a_len, _ = a_trg.shape

        # ---- Auxiliary branch ------------------------------------------------
        a_enc_out, (a_hidden, a_cell) = self.a_encoder(a_src)
        a_hidden_dec = self.a_projection(a_hidden)
        a_cell_dec = self.a_projection(a_cell)

        a_outputs = torch.zeros(batch_size, a_len, self.a_decoder.output_dim, device=self.device)
        for t in range(a_len):
            a_input = a_trg[:, t]
            a_out, a_hidden_dec, a_cell_dec = self.a_decoder(
                a_input, a_hidden_dec, a_enc_out, a_cell_dec
            )
            a_outputs[:, t] = a_out

        # ---- Target branch ---------------------------------------------------
        t_enc_out, (t_hidden, t_cell) = self.t_encoder(t_src)

        if self.transform_type == "mfa":
            # project auxiliary encoder outputs and concatenate with target
            aux_proj = self.mfa_projection(a_enc_out.transpose(1, 2))  # (B, mfa_d_model, a_len)
            feat = torch.cat((aux_proj.transpose(1, 2), t_enc_out), dim=-1)  # (B, t_len, *)
            attn_out = self.mfa(feat.transpose(1, 2))  # (B, t_len, mfa_d_model)
            summary = attn_out.mean(dim=1)  # (B, mfa_d_model)
            t_hidden_dec = self.h_proj(summary).unsqueeze(0)
            t_cell_dec = self.c_proj(summary).unsqueeze(0)
        else:
            # simple concat + linear projection
            t_hidden_dec = self.t_projection(torch.cat((t_hidden, a_hidden), dim=2))
            t_cell_dec = self.t_projection(torch.cat((t_cell, a_cell), dim=2))

        t_outputs = torch.zeros(batch_size, t_len, self.t_decoder.output_dim, device=self.device)
        for tt in range(t_len):
            t_input = t_trg[:, tt]
            t_out, t_hidden_dec, t_cell_dec = self.t_decoder(
                t_input, t_hidden_dec, t_enc_out, t_cell_dec
            )
            t_outputs[:, tt] = t_out

        return t_outputs, a_outputs

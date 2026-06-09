from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1), :]


class FluTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: Optional[int] = None,
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        dim_feedforward = dim_feedforward or 4 * d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_proj = nn.Linear(d_model, 1)

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.transformer(h, src_key_padding_mask=src_key_padding_mask)
        return self.output_proj(h).squeeze(-1)


class FluBiRNN(nn.Module):
    """Bidirectional RNN baseline for sequence-to-sequence regression."""

    def __init__(
        self,
        input_dim: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        rnn_type: str = "lstm",
        dropout: float = 0.15,
        proj_dim: Optional[int] = 128,
    ) -> None:
        super().__init__()
        rnn_type = rnn_type.lower()
        if rnn_type not in ("rnn", "gru", "lstm"):
            raise ValueError("rnn_type must be one of: rnn, gru, lstm")

        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[rnn_type]
        self.rnn = rnn_cls(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )

        out_dim = hidden_size * 2
        if proj_dim is None:
            self.head = nn.Linear(out_dim, 1)
        else:
            self.head = nn.Sequential(
                nn.Linear(out_dim, proj_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(proj_dim, 1),
            )

    def forward(
        self,
        x: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if src_key_padding_mask is not None:
            if src_key_padding_mask.dtype != torch.bool:
                src_key_padding_mask = src_key_padding_mask.bool()
            lengths = (~src_key_padding_mask).sum(dim=1).to(torch.int64).cpu()
            packed_x = nn.utils.rnn.pack_padded_sequence(
                x, lengths, batch_first=True, enforce_sorted=False
            )
            packed_h, _ = self.rnn(packed_x)
            h, _ = nn.utils.rnn.pad_packed_sequence(
                packed_h, batch_first=True, total_length=x.size(1)
            )
        else:
            h, _ = self.rnn(x)

        out = self.head(h).squeeze(-1)
        if src_key_padding_mask is not None:
            out = out.masked_fill(src_key_padding_mask, 0.0)
        return out

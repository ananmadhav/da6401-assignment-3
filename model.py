import math
import copy
import os
import gdown
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Multi30kDataset


def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:

    d_k = Q.size(-1)

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        scores = scores.masked_fill(mask, float('-inf'))

    attn_weights = torch.softmax(scores, dim=-1)

    output = torch.matmul(attn_weights, V)

    return output, attn_weights


def make_src_mask(
    src: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    return (src == pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(
    tgt: torch.Tensor,
    pad_idx: int = 1,
) -> torch.Tensor:

    batch_size, tgt_len = tgt.shape

    tgt_pad_mask = (tgt == pad_idx).unsqueeze(1).unsqueeze(2)

    causal_mask = torch.triu(
        torch.ones((tgt_len, tgt_len), device=tgt.device),
        diagonal=1
    ).bool()

    causal_mask = causal_mask.unsqueeze(0).unsqueeze(1)

    return tgt_pad_mask | causal_mask


class MultiHeadAttention(nn.Module):

    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1) -> None:
        super().__init__()

        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):

        batch_size, seq_len, d_model = x.size()

        x = x.view(
            batch_size,
            seq_len,
            self.num_heads,
            self.d_k
        )

        return x.transpose(1, 2)

    def combine_heads(self, x):

        batch_size, num_heads, seq_len, d_k = x.size()

        x = x.transpose(1, 2).contiguous()

        return x.view(
            batch_size,
            seq_len,
            self.d_model
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        Q = self.W_q(query)
        K = self.W_k(key)
        V = self.W_v(value)

        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        attention_output, attention_weights = scaled_dot_product_attention(
            Q,
            K,
            V,
            mask
        )

        attention_output = self.combine_heads(attention_output)

        output = self.W_o(attention_output)

        return output


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()

        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) *
            (-math.log(10000.0) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)

        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        seq_len = x.size(1)

        x = x + self.pe[:, :seq_len]

        return self.dropout(x)


class PositionwiseFeedForward(nn.Module):

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        x = self.linear1(x)

        x = F.relu(x)

        x = self.dropout(x)

        x = self.linear2(x)

        return x


class EncoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        raise NotImplementedError

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError


class Encoder(nn.Module):

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()

        raise NotImplementedError

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        raise NotImplementedError


class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()

        raise NotImplementedError

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size: int = None,
        tgt_vocab_size: int = None,
        d_model: int = 512,
        N: int = 6,
        num_heads: int = 8,
        d_ff: int = 2048,
        dropout: float = 0.1,
        checkpoint_path: str = None,
    ) -> None:
        super().__init__()

        dataset = Multi30kDataset(split="train")

        if src_vocab_size is None:
            src_vocab_size = len(dataset.src_vocab)

        if tgt_vocab_size is None:
            tgt_vocab_size = len(dataset.tgt_vocab)

        self.src_vocab = dataset.src_vocab
        self.tgt_vocab = dataset.tgt_vocab

        self.src_itos = dataset.src_itos
        self.tgt_itos = dataset.tgt_itos

        self.src_pad_idx = dataset.src_pad_idx
        self.tgt_pad_idx = dataset.tgt_pad_idx

        raise NotImplementedError

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        raise NotImplementedError

    def infer(self, src_sentence: str) -> str:

        raise NotImplementedError
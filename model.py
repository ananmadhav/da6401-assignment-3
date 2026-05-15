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

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, src_mask: torch.Tensor) -> torch.Tensor:

        attn_output = self.self_attn(x, x, x, src_mask)

        x = self.norm1(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)

        x = self.norm2(x + self.dropout(ffn_output))

        return x


class DecoderLayer(nn.Module):

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float = 0.1) -> None:
        super().__init__()

        self.self_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.cross_attn = MultiHeadAttention(d_model, num_heads, dropout)

        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        attn_output = self.self_attn(x, x, x, tgt_mask)

        x = self.norm1(x + self.dropout(attn_output))

        attn_output = self.cross_attn(x, memory, memory, src_mask)

        x = self.norm2(x + self.dropout(attn_output))

        ffn_output = self.ffn(x)

        x = self.norm3(x + self.dropout(ffn_output))

        return x

class Encoder(nn.Module):

    def __init__(self, layer: EncoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, mask)

        return self.norm(x)


class Decoder(nn.Module):

    def __init__(self, layer: DecoderLayer, N: int) -> None:
        super().__init__()

        self.layers = nn.ModuleList(
            [copy.deepcopy(layer) for _ in range(N)]
        )

        self.norm = nn.LayerNorm(layer.self_attn.d_model)

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)

        return self.norm(x)


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

        self.dataset = dataset

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

        self.src_embedding = nn.Embedding(
            src_vocab_size,
            d_model,
            padding_idx=self.src_pad_idx
        )

        self.tgt_embedding = nn.Embedding(
            tgt_vocab_size,
            d_model,
            padding_idx=self.tgt_pad_idx
        )

        self.positional_encoding = PositionalEncoding(
            d_model,
            dropout
        )

        encoder_layer = EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        decoder_layer = DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder = Encoder(
            encoder_layer,
            N
        )

        self.decoder = Decoder(
            decoder_layer,
            N
        )

        self.output_layer = nn.Linear(
            d_model,
            tgt_vocab_size
        )

        self.d_model = d_model

        checkpoint_path="checkpoint.pt"

        if not os.path.exists(
            checkpoint_path
        ):

            gdown.download(
                id="12LFjvW0gHBgiFCUSn25FgsflJOaq57yf",
                output=checkpoint_path,
                quiet=False
            )

        checkpoint=torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        self.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
    ) -> torch.Tensor:

        x = self.src_embedding(
            src
        ) * math.sqrt(
            self.d_model
        )

        x=self.positional_encoding(x)

        return self.encoder(
            x,
            src_mask
        )

    def decode(
        self,
        memory: torch.Tensor,
        src_mask: torch.Tensor,
        tgt: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        x=self.tgt_embedding(
            tgt
        ) * math.sqrt(
            self.d_model
        )

        x=self.positional_encoding(
            x
        )

        x=self.decoder(
            x,
            memory,
            src_mask,
            tgt_mask
        )

        return self.output_layer(x)

    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: torch.Tensor,
        tgt_mask: torch.Tensor,
    ) -> torch.Tensor:

        memory=self.encode(
            src,
            src_mask
        )

        return self.decode(
            memory,
            src_mask,
            tgt,
            tgt_mask
        )

    def infer(
        self,
        src_sentence:str
    ):

        self.eval()

        tokens=["<sos>"]

        tokens += [
            t.text.lower()
            for t in self.dataset.de_tokenizer(
                src_sentence
            )
        ]

        tokens += ["<eos>"]

        src_indices=[]

        for token in tokens:

            src_indices.append(
                self.src_vocab.get(
                    token,
                    self.src_vocab["<unk>"]
                )
            )

        src=torch.tensor(
            src_indices,
            dtype=torch.long
        ).unsqueeze(0)

        device=next(
            self.parameters()
        ).device

        src=src.to(device)

        src_mask=make_src_mask(
            src,
            self.src_pad_idx
        )

        from train import greedy_decode

        prediction=greedy_decode(
            self,
            src,
            src_mask,
            max_len=100,
            start_symbol=self.tgt_vocab["<sos>"],
            end_symbol=self.tgt_vocab["<eos>"],
            device=device
        )

        output=[]

        for idx in prediction[0]:

            word=self.tgt_itos[
                idx.item()
            ]

            if word in [
                "<sos>",
                "<eos>",
                "<pad>"
            ]:
                continue

            output.append(word)

        return " ".join(output)
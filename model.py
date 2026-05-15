import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from dataset import Multi30kDataset


def make_src_mask(src,pad_idx):

    return (src!=pad_idx).unsqueeze(1).unsqueeze(2)


def make_tgt_mask(tgt,pad_idx):

    tgt_pad_mask=(tgt!=pad_idx).unsqueeze(1).unsqueeze(2)

    seq_len=tgt.shape[1]

    causal_mask=torch.tril(

        torch.ones(
            seq_len,
            seq_len,
            device=tgt.device
        )

    ).bool()

    causal_mask=causal_mask.unsqueeze(0).unsqueeze(1)

    return tgt_pad_mask & causal_mask


class MultiHeadAttention(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads
    ):

        super().__init__()

        assert d_model % num_heads == 0

        self.d_model=d_model

        self.num_heads=num_heads

        self.head_dim=d_model//num_heads

        self.W_q=nn.Linear(
            d_model,
            d_model
        )

        self.W_k=nn.Linear(
            d_model,
            d_model
        )

        self.W_v=nn.Linear(
            d_model,
            d_model
        )

        self.fc=nn.Linear(
            d_model,
            d_model
        )

    def forward(
        self,
        query,
        key,
        value,
        mask=None
    ):

        batch_size=query.shape[0]

        Q=self.W_q(query)

        K=self.W_k(key)

        V=self.W_v(value)

        Q=Q.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K=K.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V=V.view(
            batch_size,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        scores=torch.matmul(

            Q,

            K.transpose(-2,-1)

        )/math.sqrt(
            self.head_dim
        )

        if mask is not None:

            scores=scores.masked_fill(

                mask==0,

                torch.finfo(
                    scores.dtype
                ).min

            )

        attention=F.softmax(
            scores,
            dim=-1
        )

        out=torch.matmul(
            attention,
            V
        )

        out=out.transpose(
            1,
            2
        ).contiguous()

        out=out.view(
            batch_size,
            -1,
            self.d_model
        )

        return self.fc(out)


class PositionalEncoding(nn.Module):

    def __init__(
        self,
        d_model,
        dropout=0.1,
        max_len=5000
    ):

        super().__init__()

        self.dropout=nn.Dropout(
            dropout
        )

        pe=torch.zeros(
            max_len,
            d_model
        )

        position=torch.arange(
            0,
            max_len
        ).unsqueeze(1)

        div_term=torch.exp(

            torch.arange(
                0,
                d_model,
                2
            )

            *

            (-math.log(10000.0)/d_model)

        )

        pe[:,0::2]=torch.sin(
            position*div_term
        )

        pe[:,1::2]=torch.cos(
            position*div_term
        )

        pe=pe.unsqueeze(0)

        self.register_buffer(
            "pe",
            pe
        )

    def forward(
        self,
        x
    ):

        x=x+self.pe[
            :,
            :x.size(1)
        ]

        return self.dropout(x)


class FeedForward(nn.Module):

    def __init__(
        self,
        d_model,
        d_ff,
        dropout
    ):

        super().__init__()

        self.net=nn.Sequential(

            nn.Linear(
                d_model,
                d_ff
            ),

            nn.ReLU(),

            nn.Dropout(
                dropout
            ),

            nn.Linear(
                d_ff,
                d_model
            )

        )

    def forward(
        self,
        x
    ):

        return self.net(x)


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attn=MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ff=FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1=nn.LayerNorm(
            d_model
        )

        self.norm2=nn.LayerNorm(
            d_model
        )

        self.dropout=nn.Dropout(
            dropout
        )

    def forward(
        self,
        x,
        mask
    ):

        attn=self.self_attn(
            x,
            x,
            x,
            mask
        )

        x=self.norm1(
            x+self.dropout(attn)
        )

        ff=self.ff(x)

        x=self.norm2(
            x+self.dropout(ff)
        )

        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        num_heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attn=MultiHeadAttention(
            d_model,
            num_heads
        )

        self.cross_attn=MultiHeadAttention(
            d_model,
            num_heads
        )

        self.ff=FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1=nn.LayerNorm(d_model)

        self.norm2=nn.LayerNorm(d_model)

        self.norm3=nn.LayerNorm(d_model)

        self.dropout=nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask
    ):

        attn=self.self_attn(
            x,
            x,
            x,
            tgt_mask
        )

        x=self.norm1(
            x+self.dropout(attn)
        )

        attn=self.cross_attn(
            x,
            memory,
            memory,
            src_mask
        )

        x=self.norm2(
            x+self.dropout(attn)
        )

        ff=self.ff(x)

        x=self.norm3(
            x+self.dropout(ff)
        )

        return x


class Encoder(nn.Module):

    def __init__(
        self,
        layer,
        N
    ):

        super().__init__()

        self.layers=nn.ModuleList(
            [layer for _ in range(N)]
        )

    def forward(
        self,
        x,
        mask
    ):

        for layer in self.layers:

            x=layer(
                x,
                mask
            )

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        layer,
        N
    ):

        super().__init__()

        self.layers=nn.ModuleList(
            [layer for _ in range(N)]
        )

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask
    ):

        for layer in self.layers:

            x=layer(
                x,
                memory,
                src_mask,
                tgt_mask
            )

        return x


class Transformer(nn.Module):

    def __init__(
        self,
        src_vocab_size=None,
        tgt_vocab_size=None,
        d_model=512,
        N=6,
        num_heads=8,
        d_ff=2048,
        dropout=0.1,
        checkpoint_path=None
    ):

        super().__init__()

        dataset=Multi30kDataset(
            split="train"
        )

        self.dataset=dataset

        self.src_vocab=dataset.src_vocab
        self.tgt_vocab=dataset.tgt_vocab

        self.src_pad_idx=dataset.src_vocab["<pad>"]
        self.tgt_pad_idx=dataset.tgt_vocab["<pad>"]

        if src_vocab_size is None:
            src_vocab_size=len(
                self.src_vocab
            )

        if tgt_vocab_size is None:
            tgt_vocab_size=len(
                self.tgt_vocab
            )

        self.d_model=d_model

        self.src_embedding=nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding=nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        self.positional_encoding=PositionalEncoding(
            d_model,
            dropout
        )

        enc=EncoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        dec=DecoderLayer(
            d_model,
            num_heads,
            d_ff,
            dropout
        )

        self.encoder=Encoder(
            enc,
            N
        )

        self.decoder=Decoder(
            dec,
            N
        )

        self.output_layer=nn.Linear(
            d_model,
            tgt_vocab_size
        )

    def encode(
        self,
        src,
        src_mask
    ):

        x=self.src_embedding(
            src
        )*math.sqrt(
            self.d_model
        )

        x=self.positional_encoding(
            x
        )

        return self.encoder(
            x,
            src_mask
        )

    def decode(
        self,
        memory,
        src_mask,
        tgt,
        tgt_mask
    ):

        x=self.tgt_embedding(
            tgt
        )*math.sqrt(
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

        return self.output_layer(
            x
        )

    def forward(
        self,
        src,
        tgt,
        src_mask,
        tgt_mask
    ):

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
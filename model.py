import math
import os
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import gdown

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

    def __init__(self,d_model,num_heads):

        super().__init__()

        assert d_model % num_heads==0

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

        B=query.size(0)

        Q=self.W_q(query)
        K=self.W_k(key)
        V=self.W_v(value)

        Q=Q.view(
            B,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        K=K.view(
            B,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        V=V.view(
            B,
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1,2)

        scores=torch.matmul(
            Q,
            K.transpose(-2,-1)
        )

        scores=scores/math.sqrt(
            self.head_dim
        )

        # IMPORTANT:
        # force fp32 before masking
        # prevents AMP/FP16 overflow
        scores=scores.float()

        if mask is not None:

            if mask.dim()==3:
                mask=mask.unsqueeze(1)

            mask=mask.bool()

            scores=scores.masked_fill(
                ~mask,
                -1e9
            )

        attention=F.softmax(
            scores,
            dim=-1
        )

        attention=attention.type_as(
            V
        )

        out=torch.matmul(
            attention,
            V
        )

        out=out.transpose(
            1,
            2
        )

        out=out.contiguous()

        out=out.view(
            B,
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
            max_len
        ).unsqueeze(1)

        div_term=torch.exp(
            torch.arange(
                0,
                d_model,
                2
            )*
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

    def forward(self,x):

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
            nn.Dropout(dropout),
            nn.Linear(
                d_ff,
                d_model
            )
        )

    def forward(self,x):
        return self.net(x)


class EncoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attn=MultiHeadAttention(
            d_model,
            heads
        )

        self.ff=FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)

        self.drop=nn.Dropout(dropout)

    def forward(self,x,mask):

        x=self.norm1(
            x+self.drop(
                self.self_attn(
                    x,x,x,mask
                )
            )
        )

        x=self.norm2(
            x+self.drop(
                self.ff(x)
            )
        )

        return x


class DecoderLayer(nn.Module):

    def __init__(
        self,
        d_model,
        heads,
        d_ff,
        dropout
    ):

        super().__init__()

        self.self_attn=MultiHeadAttention(
            d_model,
            heads
        )

        self.cross_attn=MultiHeadAttention(
            d_model,
            heads
        )

        self.ff=FeedForward(
            d_model,
            d_ff,
            dropout
        )

        self.norm1=nn.LayerNorm(d_model)
        self.norm2=nn.LayerNorm(d_model)
        self.norm3=nn.LayerNorm(d_model)

        self.drop=nn.Dropout(dropout)

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask
    ):

        x=self.norm1(
            x+self.drop(
                self.self_attn(
                    x,x,x,tgt_mask
                )
            )
        )

        x=self.norm2(
            x+self.drop(
                self.cross_attn(
                    x,
                    memory,
                    memory,
                    src_mask
                )
            )
        )

        x=self.norm3(
            x+self.drop(
                self.ff(x)
            )
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
            [
                copy.deepcopy(layer)
                for _ in range(N)
            ]
        )

    def forward(self,x,mask):

        for l in self.layers:
            x=l(x,mask)

        return x


class Decoder(nn.Module):

    def __init__(
        self,
        layer,
        N
    ):

        super().__init__()

        self.layers=nn.ModuleList(
            [
                copy.deepcopy(layer)
                for _ in range(N)
            ]
        )

    def forward(
        self,
        x,
        memory,
        src_mask,
        tgt_mask
    ):

        for l in self.layers:

            x=l(
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
        checkpoint_path="checkpoint.pt"
    ):

        super().__init__()

        dataset=Multi30kDataset(
            split="train"
        )

        self.dataset=dataset

        self.src_vocab=dataset.src_vocab
        self.tgt_vocab=dataset.tgt_vocab

        self.src_itos=dataset.src_itos
        self.tgt_itos=dataset.tgt_itos

        self.src_pad_idx=self.src_vocab["<pad>"]
        self.tgt_pad_idx=self.tgt_vocab["<pad>"]

        if src_vocab_size is None:
            src_vocab_size=len(
                self.src_vocab
            )

        if tgt_vocab_size is None:
            tgt_vocab_size=len(
                self.tgt_vocab
            )

        self.src_embedding=nn.Embedding(
            src_vocab_size,
            d_model
        )

        self.tgt_embedding=nn.Embedding(
            tgt_vocab_size,
            d_model
        )

        self.pos=PositionalEncoding(
            d_model
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

        if checkpoint_path:

            if not os.path.exists(
                checkpoint_path
            ):

                gdown.download(
                    "https://drive.google.com/uc?id=12LFjvW0gHBgiFCUSn25FgsflJOaq57yf",
                    checkpoint_path,
                    quiet=False
                )

            state=torch.load(
                checkpoint_path,
                map_location="cpu"
            )

            self.load_state_dict(
                state["model_state_dict"],
                strict=False
            )

    def encode(
        self,
        src,
        src_mask
    ):

        x=self.src_embedding(src)

        x=self.pos(x)

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
        )

        x=self.pos(x)

        x=self.decoder(
            x,
            memory,
            src_mask,
            tgt_mask
        )

        return self.output_layer(x)

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

    def infer(
        self,
        german_sentence
    ):

        from train import greedy_decode

        self.eval()

        tokens=["<sos>"]

        tokens+=[
            t.text.lower()
            for t in
            self.dataset.de_tokenizer(
                german_sentence
            )
        ]

        tokens+=["<eos>"]

        ids=[

            self.src_vocab.get(
                t,
                self.src_vocab["<unk>"]
            )

            for t in tokens

        ]

        src=torch.tensor(
            ids
        ).unsqueeze(0)

        device=next(
            self.parameters()
        ).device

        src=src.to(device)

        src_mask=make_src_mask(
            src,
            self.src_pad_idx
        )

        pred=greedy_decode(
            self,
            src,
            src_mask,
            40,
            self.tgt_vocab["<sos>"],
            self.tgt_vocab["<eos>"],
            device
        )

        words=[]

        for idx in pred[0]:

            w=self.tgt_itos[
                idx.item()
            ]

            if w in [
                "<sos>",
                "<eos>",
                "<pad>"
            ]:
                continue

            words.append(w)

        return " ".join(words)
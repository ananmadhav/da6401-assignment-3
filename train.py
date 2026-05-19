import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from nltk.translate.bleu_score import corpus_bleu

from model import Transformer
from model import make_src_mask
from model import make_tgt_mask

from dataset import (
    Multi30kDataset,
    collate_fn
)

from lr_scheduler import (
    NoamScheduler
)


class LabelSmoothingLoss(
    nn.Module
):

    def __init__(
        self,
        vocab_size,
        pad_idx,
        smoothing=0.1
    ):

        super().__init__()

        self.criterion=(
            nn.CrossEntropyLoss(
                ignore_index=pad_idx,
                label_smoothing=smoothing
            )
        )

    def forward(
        self,
        logits,
        target
    ):

        return self.criterion(
            logits,
            target
        )


def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    end_symbol,
    device
):

    ys=torch.ones(
        1,
        1
    ).fill_(
        start_symbol
    ).type(torch.long).to(device)

    memory=model.encode(
        src,
        src_mask
    )

    for i in range(max_len):

        tgt_mask=make_tgt_mask(
            ys,
            model.tgt_pad_idx
        )

        out=model.decode(
            memory,
            src_mask,
            ys,
            tgt_mask
        )

        prob=out[:,-1]

        _,next_word=torch.max(
            prob,
            dim=1
        )

        next_word=next_word.item()

        ys=torch.cat(

            [
                ys,

                torch.ones(
                    1,
                    1
                ).type_as(src).fill_(
                    next_word
                )
            ],

            dim=1
        )

        if next_word==end_symbol:
            break

    return ys


def compute_bleu(
    model,
    val_loader,
    device
):

    model.eval()

    references=[]
    hypotheses=[]

    with torch.no_grad():

        for src,tgt in tqdm(
            val_loader
        ):

            src=src.to(device)

            for i in range(src.size(0)):

                src_ex=src[i].unsqueeze(0)

                src_mask=make_src_mask(
                    src_ex,
                    model.src_pad_idx
                )

                pred=greedy_decode(

                    model,

                    src_ex,

                    src_mask,

                    40,

                    model.tgt_vocab["<sos>"],

                    model.tgt_vocab["<eos>"],

                    device
                )

                pred_words=[]

                for idx in pred[0]:

                    w=model.dataset.tgt_itos[
                        idx.item()
                    ]

                    if w in [

                        "<sos>",
                        "<eos>",
                        "<pad>"

                    ]:
                        continue

                    pred_words.append(w)

                tgt_words=[]

                for idx in tgt[i]:

                    w=model.dataset.tgt_itos[
                        idx.item()
                    ]

                    if w in [

                        "<sos>",
                        "<eos>",
                        "<pad>"

                    ]:
                        continue

                    tgt_words.append(w)

                hypotheses.append(
                    pred_words
                )

                references.append(
                    [tgt_words]
                )

    return corpus_bleu(
        references,
        hypotheses
    )


def run_epoch(

    data_iter,

    model,

    loss_fn,

    optimizer,

    scheduler=None,

    epoch_num=0,

    is_train=True,

    device="cpu"

):

    if is_train:
        model.train()

    else:
        model.eval()

    total_loss=0
    total_correct=0
    total_tokens=0

    loop=tqdm(
        data_iter
    )

    for step,(src,tgt) in enumerate(
        loop
    ):

        src=src.to(device)

        tgt=tgt.to(device)

        tgt_input=tgt[:,:-1]

        targets=tgt[:,1:]

        src_mask=make_src_mask(
            src,
            model.src_pad_idx
        )

        tgt_mask=make_tgt_mask(
            tgt_input,
            model.tgt_pad_idx
        )

        if is_train:
            optimizer.zero_grad()

        with torch.set_grad_enabled(
            is_train
        ):

            logits=model(

                src,

                tgt_input,

                src_mask,

                tgt_mask

            )

            logits=logits.reshape(
                -1,
                logits.shape[-1]
            )

            targets=targets.reshape(
                -1
            )

            loss=loss_fn(
                logits,
                targets
            )

            predictions=torch.argmax(
                logits,
                dim=1
            )

            mask=(
                targets
                !=
                model.tgt_pad_idx
            )

            correct=(

                predictions[
                    mask
                ]

                ==
                targets[
                    mask
                ]

            ).sum()

            total_correct+=(
                correct.item()
            )

            total_tokens+=(
                mask.sum().item()
            )

            if is_train:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=1.0

                )

                optimizer.step()

                if scheduler:
                    scheduler.step()

        total_loss+=loss.item()

    avg_loss=(
        total_loss/
        len(data_iter)
    )

    accuracy=(
        total_correct/
        total_tokens
    )

    return avg_loss,accuracy


def run_training_experiment():

    config={

        "experiment_name":
        "LearnedPositional",

        "use_scaling":
        True,

        "use_learned_pos":
        True,

        "epochs":
        10,

        "batch_size":
        64,

        "d_model":
        512,

        "warmup":
        4000
    }

    wandb.init(

        project=
        "da6401-a3",

        name=
        config[
            "experiment_name"
        ],

        config=config
    )

    device=(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    train_dataset=Multi30kDataset(
        split="train"
    )

    val_dataset=Multi30kDataset(
        split="validation"
    )

    train_loader=DataLoader(

        train_dataset,

        batch_size=
        config[
            "batch_size"
        ],

        shuffle=True,

        collate_fn=
        collate_fn
    )

    val_loader=DataLoader(

        val_dataset,

        batch_size=
        config[
            "batch_size"
        ],

        collate_fn=
        collate_fn
    )

    model=Transformer(

        checkpoint_path=None,

        use_scaling=
        config[
            "use_scaling"
        ],

        use_learned_pos=
        config[
            "use_learned_pos"
        ]

    ).to(device)

    optimizer=torch.optim.Adam(

        model.parameters(),

        lr=1,

        betas=
        (
            0.9,
            0.98
        ),

        eps=1e-9
    )

    scheduler=NoamScheduler(

        optimizer,

        d_model=
        config[
            "d_model"
        ],

        warmup_steps=
        config[
            "warmup"
        ]
    )

    loss_fn=LabelSmoothingLoss(

        len(
            model.tgt_vocab
        ),

        model.tgt_pad_idx
    )

    for epoch in range(
        config["epochs"]
    ):

        train_loss,train_acc=run_epoch(

            train_loader,

            model,

            loss_fn,

            optimizer,

            scheduler,

            epoch,

            True,

            device
        )

        val_loss,val_acc=run_epoch(

            val_loader,

            model,

            loss_fn,

            None,

            None,

            epoch,

            False,

            device
        )

        bleu=compute_bleu(
            model,
            val_loader,
            device
        )

        wandb.log({

            "train_loss":
            train_loss,

            "val_loss":
            val_loss,

            "train_accuracy":
            train_acc,

            "val_accuracy":
            val_acc,

            "BLEU":
            bleu

        })

    wandb.finish()


if __name__=="__main__":

    run_training_experiment()
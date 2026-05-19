import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

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

                ignore_index=
                pad_idx,

                label_smoothing=
                smoothing
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

        src=src.to(
            device,
            non_blocking=True
        )

        tgt=tgt.to(
            device,
            non_blocking=True
        )

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

            with torch.amp.autocast(
                "cuda"
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

                global_step=(

                    epoch_num
                    *
                    len(data_iter)

                    +

                    step

                )

                if global_step<1000:

                    q_grad=(

                        model
                        .encoder
                        .layers[0]
                        .self_attn
                        .W_q
                        .weight
                        .grad
                        .norm()
                        .item()

                    )

                    k_grad=(

                        model
                        .encoder
                        .layers[0]
                        .self_attn
                        .W_k
                        .weight
                        .grad
                        .norm()
                        .item()

                    )

                    wandb.log({

                        "Q_gradient_norm":
                        q_grad,

                        "K_gradient_norm":
                        k_grad,

                        "global_step":
                        global_step

                    })

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=1.0

                )

                optimizer.step()

                if scheduler:

                    scheduler.step()

                wandb.log({

                    "step_train_loss":
                    loss.item(),

                    "learning_rate":
                    optimizer.param_groups[0]["lr"]

                })

        total_loss+=loss.item()

        loop.set_description(
            f"Epoch {epoch_num}"
        )

        loop.set_postfix(

            avg_loss=

            total_loss/

            (loop.n+1)

        )

    avg_loss=(
        total_loss/
        len(data_iter)
    )

    accuracy=(

        total_correct

        /

        total_tokens

    )

    return (
        avg_loss,
        accuracy
    )


def save_checkpoint(

    model,

    optimizer,

    scheduler,

    epoch,

    path="checkpoint.pt"

):

    state={

        "epoch":
        epoch,

        "model_state_dict":
        model.state_dict(),

        "optimizer_state_dict":
        optimizer.state_dict()

    }

    if scheduler:

        state[
            "scheduler_state_dict"
        ]=scheduler.state_dict()

    torch.save(
        state,
        path
    )


def run_training_experiment():

    config={

        ################################

        "experiment_name":
        "NoScaling",

        "use_scaling":
        False,

        ################################

        "epochs":
        10,

        "batch_size":
        64,

        "d_model":
        512,

        "warmup":
        4000,

        "label_smoothing":
        0.1
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

    train_dataset=(
        Multi30kDataset(
            split="train"
        )
    )

    val_dataset=(
        Multi30kDataset(
            split="validation"
        )
    )

    train_loader=DataLoader(

        train_dataset,

        batch_size=
        config[
            "batch_size"
        ],

        shuffle=True,

        collate_fn=
        collate_fn,

        num_workers=4,

        pin_memory=True
    )

    val_loader=DataLoader(

        val_dataset,

        batch_size=
        config[
            "batch_size"
        ],

        collate_fn=
        collate_fn,

        num_workers=4,

        pin_memory=True
    )

    model=Transformer(

        checkpoint_path=None,

        use_scaling=
        config[
            "use_scaling"
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

    loss_fn=(
        LabelSmoothingLoss(

            len(
                model.tgt_vocab
            ),

            model.tgt_pad_idx,

            config[
                "label_smoothing"
            ]
        )
    )

    best_val=float(
        "inf"
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

        perplexity=torch.exp(
            torch.tensor(
                val_loss
            )
        ).item()

        wandb.log({

            "epoch":
            epoch,

            "train_loss":
            train_loss,

            "train_accuracy":
            train_acc,

            "val_loss":
            val_loss,

            "val_accuracy":
            val_acc,

            "perplexity":
            perplexity
        })

        if val_loss<best_val:

            best_val=val_loss

            save_checkpoint(

                model,

                optimizer,

                scheduler,

                epoch
            )

    wandb.finish()


if __name__=="__main__":

    run_training_experiment()
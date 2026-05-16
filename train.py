import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset, collate_fn
from lr_scheduler import NoamScheduler


class LabelSmoothingLoss(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_idx,
        smoothing=0.05
    ):
        super().__init__()

        self.criterion=nn.CrossEntropyLoss(
            ignore_index=pad_idx,
            label_smoothing=smoothing
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

    loop=tqdm(data_iter)

    for src,tgt in loop:

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

        with torch.amp.autocast("cuda"):

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

        if torch.isnan(loss):

            print(
                "NaN detected"
            )

            continue

        if is_train:

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            if scheduler:
                scheduler.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch_num}"
        )

        loop.set_postfix(
            avg_loss=
            total_loss/(loop.n+1)
        )

    return total_loss/len(
        data_iter
    )


def greedy_decode(
    model,
    src,
    src_mask,
    max_len,
    start_symbol,
    end_symbol,
    device="cpu"
):

    memory=model.encode(
        src,
        src_mask
    )

    ys=torch.ones(
        1,
        1,
        dtype=torch.long,
        device=device
    ).fill_(
        start_symbol
    )

    model.eval()

    with torch.no_grad():

        for _ in range(max_len):

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

            next_word=torch.argmax(
                out[:,-1],
                dim=1
            ).item()

            if next_word in [

                end_symbol,

                model.tgt_vocab["<pad>"]

            ]:
                break

            ys=torch.cat(
                [
                    ys,
                    torch.tensor(
                        [[next_word]],
                        device=device
                    )
                ],
                dim=1
            )

    return ys


def save_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch,
    path="checkpoint.pt"
):

    torch.save(
        {
            "epoch":epoch,

            "model_state_dict":
            model.state_dict(),

            "optimizer_state_dict":
            optimizer.state_dict(),

            "scheduler_state_dict":
            scheduler.state_dict()

        },
        path
    )


def run_training_experiment():

    wandb.init(
        project="da6401-a3"
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
        batch_size=64,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    val_loader=DataLoader(
        val_dataset,
        batch_size=64,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True
    )

    model=Transformer(
        checkpoint_path=None
    ).to(device)

    optimizer=torch.optim.Adam(
        model.parameters(),
        lr=1,
        betas=(0.9,0.98),
        eps=1e-9
    )

    scheduler=NoamScheduler(
        optimizer,
        d_model=512,
        warmup_steps=4000
    )

    loss_fn=LabelSmoothingLoss(
        len(model.tgt_vocab),
        model.tgt_pad_idx
    )

    best_val=float(
        "inf"
    )

    for epoch in range(30):

        train_loss=run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device
        )

        val_loss=run_epoch(
            val_loader,
            model,
            loss_fn,
            None,
            None,
            epoch,
            False,
            device
        )

        wandb.log({
            "train_loss":
            train_loss,

            "val_loss":
            val_loss
        })

        if val_loss<best_val:

            best_val=val_loss

            save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch
            )

            print(
                f"Best checkpoint saved at epoch {epoch}"
            )


if __name__=="__main__":

    run_training_experiment()
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Optional
from nltk.translate.bleu_score import corpus_bleu
from tqdm import tqdm
import wandb

from model import Transformer, make_src_mask, make_tgt_mask
from dataset import Multi30kDataset
from lr_scheduler import NoamScheduler


class LabelSmoothingLoss(nn.Module):

    def __init__(self, vocab_size: int, pad_idx: int, smoothing: float = 0.1) -> None:
        super().__init__()

        self.vocab_size = vocab_size
        self.pad_idx = pad_idx
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:

        logits = F.log_softmax(logits, dim=-1)

        with torch.no_grad():

            true_dist = torch.zeros_like(logits)

            true_dist.fill_(
                self.smoothing / (self.vocab_size - 2)
            )

            true_dist.scatter_(
                1,
                target.unsqueeze(1),
                self.confidence
            )

            true_dist[:, self.pad_idx] = 0

            mask = target == self.pad_idx

            true_dist[mask] = 0

        return torch.mean(
            torch.sum(-true_dist * logits, dim=1)
        )


def run_epoch(
    data_iter,
    model: Transformer,
    loss_fn: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    scheduler=None,
    epoch_num: int = 0,
    is_train: bool = True,
    device: str = "cpu",
) -> float:

    if is_train:
        model.train()

    else:
        model.eval()

    total_loss = 0

    loop = tqdm(data_iter)

    for src, tgt in loop:

        src = src.to(device)
        tgt = tgt.to(device)

        tgt_input = tgt[:, :-1]

        targets = tgt[:, 1:]

        src_mask = make_src_mask(
            src,
            model.src_pad_idx
        )

        tgt_mask = make_tgt_mask(
            tgt_input,
            model.tgt_pad_idx
        )

        logits = model(
            src,
            tgt_input,
            src_mask,
            tgt_mask
        )

        logits = logits.reshape(
            -1,
            logits.shape[-1]
        )

        targets = targets.reshape(-1)

        loss = loss_fn(
            logits,
            targets
        )

        if is_train:

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            if scheduler is not None:
                scheduler.step()

        total_loss += loss.item()

        loop.set_description(
            f"Epoch {epoch_num}"
        )

        loop.set_postfix(
            loss=loss.item()
        )

    return total_loss / len(data_iter)


def greedy_decode(
    model: Transformer,
    src: torch.Tensor,
    src_mask: torch.Tensor,
    max_len: int,
    start_symbol: int,
    end_symbol: int,
    device: str = "cpu",
) -> torch.Tensor:

    memory = model.encode(
        src,
        src_mask
    )

    ys = torch.ones(
        1,
        1,
        dtype=torch.long,
        device=device
    ).fill_(start_symbol)

    for _ in range(max_len - 1):

        tgt_mask = make_tgt_mask(
            ys,
            model.tgt_pad_idx
        )

        out = model.decode(
            memory,
            src_mask,
            ys,
            tgt_mask
        )

        prob = out[:, -1]

        next_word = torch.argmax(
            prob,
            dim=1
        ).item()

        ys = torch.cat(
            [
                ys,
                torch.ones(
                    1,
                    1,
                    dtype=torch.long,
                    device=device
                ).fill_(next_word)
            ],
            dim=1
        )

        if next_word == end_symbol:
            break

    return ys


def evaluate_bleu(
    model: Transformer,
    test_dataloader: DataLoader,
    tgt_vocab,
    device: str = "cpu",
    max_len: int = 100,
) -> float:

    model.eval()

    refs = []
    hyps = []

    with torch.no_grad():

        for src, tgt in tqdm(test_dataloader):

            src = src.to(device)

            for i in range(src.size(0)):

                single_src = src[i].unsqueeze(0)

                src_mask = make_src_mask(
                    single_src,
                    model.src_pad_idx
                )

                prediction = greedy_decode(
                    model,
                    single_src,
                    src_mask,
                    max_len,
                    model.tgt_vocab["<sos>"],
                    model.tgt_vocab["<eos>"],
                    device
                )

                pred_tokens = []

                for idx in prediction[0]:

                    token = model.tgt_itos[
                        idx.item()
                    ]

                    if token in [
                        "<sos>",
                        "<eos>",
                        "<pad>"
                    ]:
                        continue

                    pred_tokens.append(token)

                tgt_tokens = []

                for idx in tgt[i]:

                    token = model.tgt_itos[
                        idx.item()
                    ]

                    if token in [
                        "<sos>",
                        "<eos>",
                        "<pad>"
                    ]:
                        continue

                    tgt_tokens.append(token)

                refs.append([tgt_tokens])

                hyps.append(pred_tokens)

    return corpus_bleu(
        refs,
        hyps
    ) * 100


def save_checkpoint(
    model: Transformer,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epoch: int,
    path: str = "checkpoint.pt",
) -> None:

    torch.save(
        {
            'epoch': epoch,

            'model_state_dict':
                model.state_dict(),

            'optimizer_state_dict':
                optimizer.state_dict(),

            'scheduler_state_dict':
                scheduler.state_dict(),

            'model_config': {

                'src_vocab_size':
                    len(model.src_vocab),

                'tgt_vocab_size':
                    len(model.tgt_vocab),

                'd_model':
                    model.d_model
            }
        },
        path
    )


def load_checkpoint(
    path: str,
    model: Transformer,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
) -> int:

    checkpoint = torch.load(
        path,
        map_location="cpu"
    )

    model.load_state_dict(
        checkpoint['model_state_dict']
    )

    if optimizer is not None:

        optimizer.load_state_dict(
            checkpoint[
                'optimizer_state_dict'
            ]
        )

    if scheduler is not None:

        scheduler.load_state_dict(
            checkpoint[
                'scheduler_state_dict'
            ]
        )

    return checkpoint['epoch']


def run_training_experiment():

    wandb.init(
        project="da6401-a3"
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dataset = Multi30kDataset(
        split="train"
    )

    val_dataset = Multi30kDataset(
        split="validation"
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=32,
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=32
    )

    model = Transformer().to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=1,
        betas=(0.9,0.98),
        eps=1e-9
    )

    scheduler = NoamScheduler(
        optimizer,
        d_model=512,
        warmup_steps=4000
    )

    loss_fn = LabelSmoothingLoss(
        len(model.tgt_vocab),
        model.tgt_pad_idx
    )

    for epoch in range(20):

        train_loss = run_epoch(
            train_loader,
            model,
            loss_fn,
            optimizer,
            scheduler,
            epoch,
            True,
            device
        )

        val_loss = run_epoch(
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
            "train_loss": train_loss,
            "val_loss": val_loss
        })

        save_checkpoint(
            model,
            optimizer,
            scheduler,
            epoch
        )


if __name__ == "__main__":
    run_training_experiment()
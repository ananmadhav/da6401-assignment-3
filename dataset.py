from datasets import load_dataset
import spacy
from collections import Counter
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Multi30kDataset(Dataset):

    _dataset_cache = None
    _vocab_cache = None
    _processed_cache = {}

    def __init__(self, split='train', min_freq=2):

        self.split = split

        if Multi30kDataset._dataset_cache is None:

            Multi30kDataset._dataset_cache = load_dataset(
                "bentrevett/multi30k"
            )

        dataset = Multi30kDataset._dataset_cache

        self.dataset = dataset[split]

        try:
            self.de_tokenizer = spacy.load(
                "de_core_news_sm"
            )
        except:
            self.de_tokenizer = spacy.blank(
                "de"
            )

        try:
            self.en_tokenizer = spacy.load(
                "en_core_web_sm"
            )
        except:
            self.en_tokenizer = spacy.blank(
                "en"
            )

        self.special_tokens = [
            "<unk>",
            "<pad>",
            "<sos>",
            "<eos>"
        ]

        if Multi30kDataset._vocab_cache is None:

            self.build_vocab(min_freq)

            Multi30kDataset._vocab_cache = (

                self.src_vocab,
                self.tgt_vocab,
                self.src_itos,
                self.tgt_itos

            )

        else:

            (
                self.src_vocab,
                self.tgt_vocab,
                self.src_itos,
                self.tgt_itos

            ) = Multi30kDataset._vocab_cache

        self.src_pad_idx = self.src_vocab["<pad>"]
        self.tgt_pad_idx = self.tgt_vocab["<pad>"]

        self.src_sos_idx = self.src_vocab["<sos>"]
        self.src_eos_idx = self.src_vocab["<eos>"]

        self.tgt_sos_idx = self.tgt_vocab["<sos>"]
        self.tgt_eos_idx = self.tgt_vocab["<eos>"]

        self.src_unk_idx = self.src_vocab["<unk>"]
        self.tgt_unk_idx = self.tgt_vocab["<unk>"]

        cache_key = f"{split}_{min_freq}"

        if cache_key not in Multi30kDataset._processed_cache:

            Multi30kDataset._processed_cache[
                cache_key
            ] = self.process_data()

        self.data = Multi30kDataset._processed_cache[
            cache_key
        ]

    def tokenize_de(self, text):

        return [

            token.text.lower()

            for token in
            self.de_tokenizer(text)

        ]

    def tokenize_en(self, text):

        return [

            token.text.lower()

            for token in
            self.en_tokenizer(text)

        ]

    def build_vocab(
        self,
        min_freq=2
    ):

        src_counter = Counter()
        tgt_counter = Counter()

        train_dataset = (
            Multi30kDataset
            ._dataset_cache["train"]
        )

        for sample in train_dataset:

            src_tokens = self.tokenize_de(
                sample["de"]
            )

            tgt_tokens = self.tokenize_en(
                sample["en"]
            )

            src_counter.update(
                src_tokens
            )

            tgt_counter.update(
                tgt_tokens
            )

        self.src_vocab = {}
        self.tgt_vocab = {}

        self.src_itos = {}
        self.tgt_itos = {}

        for idx, token in enumerate(
            self.special_tokens
        ):

            self.src_vocab[token] = idx
            self.tgt_vocab[token] = idx

            self.src_itos[idx] = token
            self.tgt_itos[idx] = token

        src_idx = len(
            self.special_tokens
        )

        tgt_idx = len(
            self.special_tokens
        )

        for token, freq in src_counter.items():

            if freq >= min_freq:

                if token not in self.src_vocab:

                    self.src_vocab[
                        token
                    ] = src_idx

                    self.src_itos[
                        src_idx
                    ] = token

                    src_idx += 1

        for token, freq in tgt_counter.items():

            if freq >= min_freq:

                if token not in self.tgt_vocab:

                    self.tgt_vocab[
                        token
                    ] = tgt_idx

                    self.tgt_itos[
                        tgt_idx
                    ] = token

                    tgt_idx += 1

    def numericalize_src(
        self,
        tokens
    ):

        return [

            self.src_vocab.get(
                token,
                self.src_unk_idx
            )

            for token in tokens

        ]

    def numericalize_tgt(
        self,
        tokens
    ):

        return [

            self.tgt_vocab.get(
                token,
                self.tgt_unk_idx
            )

            for token in tokens

        ]

    def process_data(self):

        processed = []

        for sample in self.dataset:

            src_tokens = self.tokenize_de(
                sample["de"]
            )

            tgt_tokens = self.tokenize_en(
                sample["en"]
            )

            src_indices = (

                [self.src_sos_idx]

                + self.numericalize_src(
                    src_tokens
                )

                + [self.src_eos_idx]

            )

            tgt_indices = (

                [self.tgt_sos_idx]

                + self.numericalize_tgt(
                    tgt_tokens
                )

                + [self.tgt_eos_idx]

            )

            processed.append(

                (

                    torch.tensor(
                        src_indices
                    ),

                    torch.tensor(
                        tgt_indices
                    )

                )

            )

        return processed

    def __len__(self):

        return len(
            self.data
        )

    def __getitem__(
        self,
        idx
    ):

        return self.data[idx]


def collate_fn(batch):

    src_batch = []
    tgt_batch = []

    for src, tgt in batch:

        src_batch.append(src)
        tgt_batch.append(tgt)

    src_batch = pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=1
    )

    tgt_batch = pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=1
    )

    return src_batch, tgt_batch
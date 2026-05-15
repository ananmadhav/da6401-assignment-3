from datasets import load_dataset
import spacy
from collections import Counter
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class Multi30kDataset(Dataset):

    def __init__(self, split='train', min_freq=2):

        self.split = split

        dataset = load_dataset(
            "bentrevett/multi30k"
        )

        self.dataset = dataset[
            split
        ]

        try:
            self.de_tokenizer=spacy.load(
                "de_core_news_sm"
            )
        except:
            self.de_tokenizer=spacy.blank(
                "de"
            )

        try:
            self.en_tokenizer=spacy.load(
                "en_core_web_sm"
            )
        except:
            self.en_tokenizer=spacy.blank(
                "en"
            )

        self.special_tokens = [
            "<unk>",
            "<pad>",
            "<sos>",
            "<eos>"
        ]

        self.build_vocab(
            min_freq
        )

        self.process_data()

    def tokenize_de(self,text):

        return [
            token.text.lower()
            for token in
            self.de_tokenizer(text)
        ]

    def tokenize_en(self,text):

        return [
            token.text.lower()
            for token in
            self.en_tokenizer(text)
        ]

    def build_vocab(
        self,
        min_freq=2
    ):

        src_counter=Counter()
        tgt_counter=Counter()

        train_dataset=load_dataset(
            "bentrevett/multi30k"
        )["train"]

        for sample in train_dataset:

            src_tokens=self.tokenize_de(
                sample["de"]
            )

            tgt_tokens=self.tokenize_en(
                sample["en"]
            )

            src_counter.update(
                src_tokens
            )

            tgt_counter.update(
                tgt_tokens
            )

        self.src_vocab={}
        self.tgt_vocab={}

        self.src_itos={}
        self.tgt_itos={}

        for idx,token in enumerate(
            self.special_tokens
        ):

            self.src_vocab[token]=idx
            self.tgt_vocab[token]=idx

            self.src_itos[idx]=token
            self.tgt_itos[idx]=token

        src_idx=len(
            self.special_tokens
        )

        tgt_idx=len(
            self.special_tokens
        )

        for token,freq in src_counter.items():

            if freq>=min_freq:

                if token not in self.src_vocab:

                    self.src_vocab[token]=src_idx

                    self.src_itos[src_idx]=token

                    src_idx+=1

        for token,freq in tgt_counter.items():

            if freq>=min_freq:

                if token not in self.tgt_vocab:

                    self.tgt_vocab[token]=tgt_idx

                    self.tgt_itos[tgt_idx]=token

                    tgt_idx+=1

        self.src_pad_idx=self.src_vocab["<pad>"]
        self.tgt_pad_idx=self.tgt_vocab["<pad>"]

        self.src_sos_idx=self.src_vocab["<sos>"]
        self.src_eos_idx=self.src_vocab["<eos>"]

        self.tgt_sos_idx=self.tgt_vocab["<sos>"]
        self.tgt_eos_idx=self.tgt_vocab["<eos>"]

        self.src_unk_idx=self.src_vocab["<unk>"]
        self.tgt_unk_idx=self.tgt_vocab["<unk>"]

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

        self.data=[]

        for sample in self.dataset:

            src_tokens=self.tokenize_de(
                sample["de"]
            )

            tgt_tokens=self.tokenize_en(
                sample["en"]
            )

            src_indices=(
                [self.src_sos_idx]
                +self.numericalize_src(
                    src_tokens
                )
                +[self.src_eos_idx]
            )

            tgt_indices=(
                [self.tgt_sos_idx]
                +self.numericalize_tgt(
                    tgt_tokens
                )
                +[self.tgt_eos_idx]
            )

            src_tensor=torch.tensor(
                src_indices
            )

            tgt_tensor=torch.tensor(
                tgt_indices
            )

            self.data.append(
                (
                    src_tensor,
                    tgt_tensor
                )
            )

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

    src_batch=[]

    tgt_batch=[]

    for src,tgt in batch:

        src_batch.append(
            src
        )

        tgt_batch.append(
            tgt
        )

    src_batch=pad_sequence(
        src_batch,
        batch_first=True,
        padding_value=1
    )

    tgt_batch=pad_sequence(
        tgt_batch,
        batch_first=True,
        padding_value=1
    )

    return (
        src_batch,
        tgt_batch
    )
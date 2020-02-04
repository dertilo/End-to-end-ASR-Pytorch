import random

from torch.utils.data import Dataset
from corpora.spanish_corpora import spanish_corpus

train_eval_idx = None


class SpanishDataset(Dataset):
    def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
        del ascending
        assert split == "train" or split == "eval"
        # Setup
        self.path = path
        self.batch_size = bucket_size
        self.tokenizer = tokenizer

        data = list(spanish_corpus(path).items())

        global train_eval_idx
        if train_eval_idx is None:
            idx = list(range(len(data)))
            random.shuffle(idx)
            train_eval_split_idx = round(len(data) * 0.9)
            train_eval_idx = {
                "train": idx[:train_eval_split_idx],
                "eval": idx[train_eval_split_idx:],
            }

        fname_txt = [data[k] for k in train_eval_idx[split]]
        fname_txt = sorted(fname_txt, key=lambda x: len(x[1]))

        self.file_list, self.texts = zip(*[(f_name, txt) for f_name, txt in fname_txt])
        self.texts = list(self.texts[:])
        assert all([isinstance(t, str) for t in self.texts])

    def __getitem__(self, index):

        if self.batch_size > 1:
            index = min(len(self.texts) - self.batch_size, index)
            for i in range(index, index + self.batch_size):
                if type(self.texts[i]) is str:
                    self.texts[i] = self.tokenizer.encode(self.texts[i])
            return [
                (f_path, txt)
                for f_path, txt in zip(
                    self.file_list[index : index + self.batch_size],
                    self.texts[index : index + self.batch_size],
                )
            ]
        else:
            if type(self.texts[index]) is str:
                self.texts[index] = self.tokenizer.encode(self.texts[index])
            return self.file_list[index], self.texts[index]

    def __len__(self):
        return len(self.file_list)


class SpanishTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, batch_size):
        del split
        self.path = path
        self.batch_size = batch_size
        self.texts = [t for t in spanish_corpus(path).values()]
        self.tokenizer = tokenizer

    def __getitem__(self, index):
        if self.batch_size > 1:
            index = min(len(self.texts) - self.batch_size, index)
            for i in range(index, index + self.batch_size):
                if type(self.texts[i]) is str:
                    self.texts[i] = self.tokenizer.encode(self.texts[i])
            return self.texts[index : index + self.batch_size]
        else:
            if type(self.texts[index]) is str:
                self.texts[index] = self.tokenizer.encode(self.texts[index])
            return self.texts[index]

    def __len__(self):
        return len(self.texts)

import os
from tqdm import tqdm
from pathlib import Path
from os.path import join, getsize
from joblib import Parallel, delayed
from torch.utils.data import Dataset
from typing import Dict, NamedTuple
from util import data_io

# class TextAndAudioDataset(Dataset):
#     def __init__(self, path, split, tokenizer, bucket_size, ascending=False):
#         # Setup
#         self.path = path
#         self.bucket_size = bucket_size
#
#         # List all wave files
#         file_list = []
#         for s in split:
#             split_list = list(Path(join(path, s)).rglob("*.flac"))
#             assert len(split_list) > 0, "No data found @ {}".format(join(path,s))
#             file_list += split_list
#         # Read text
#         text = Parallel(n_jobs=READ_FILE_THREADS)(
#             delayed(read_text)(str(f)) for f in file_list)
#         #text = Parallel(n_jobs=-1)(delayed(tokenizer.encode)(txt) for txt in text)
#         text = [tokenizer.encode(txt) for txt in text]
#
#         # Sort dataset by text length
#         #file_len = Parallel(n_jobs=READ_FILE_THREADS)(delayed(getsize)(f) for f in file_list)
#         self.file_list, self.text = zip(*[(f_name, txt)
#                                           for f_name, txt in sorted(zip(file_list, text), reverse=not ascending, key=lambda x:len(x[1]))])
#
#     def __getitem__(self, index):
#         if self.bucket_size > 1:
#             # Return a bucket
#             index = min(len(self.file_list)-self.bucket_size, index)
#             return [(f_path, txt) for f_path, txt in
#                     zip(self.file_list[index:index+self.bucket_size], self.text[index:index+self.bucket_size])]
#         else:
#             return self.file_list[index], self.text[index]
#
#     def __len__(self):
#         return len(self.file_list)
from corpora.spanish_corpora import spanish_corpus


class SpanishTextDataset(Dataset):
    def __init__(self, path, split, tokenizer, batch_size):
        del split
        self.path = path
        self.batch_size = batch_size
        self.texts = [
            t
            for t in spanish_corpus(path).values()
        ]
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

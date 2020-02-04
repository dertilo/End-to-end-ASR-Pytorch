import argparse
import os
import random

from util import data_io

from bin.train_asr import Solver
from corpora.spanish_corpora import spanish_corpus
from corpus.spanish_dataset import TRAIN_EVAL_DATASET_FILES

if __name__ == "__main__":
    params = {
        "config": "config/spanish/asr.yaml",
        "name": None,
        "logdir": "log/",
        "ckpdir": "ckpt/",
        "outdir": "result/",
        "load": None,
        "seed": 0,
        "cudnn_ctc": False,
        "njobs": 0,
        "cpu": False,
        "no_pin": False,
        "test": False,
        "no_msg": False,
        "lm": False,
        "amp": False,
        "reserve_gpu": 0,
        "jit": False,
        "gpu": True,
        "pin_memory": True,
        "verbose": True,
    }
    # num_layers = 2 #5
    # dim = 64 #512
    num_layers = 5
    dim = 512
    config = {
        "data": {
            "corpus": {
                "name": "spanish",
                "path": "/docker-share/data/asr_datasets",
                "train_split": 'train',
                "dev_split": 'eval',
                "bucketing": True,
                "batch_size": 8,
            },
            "audio": {
                "feat_type": "fbank",
                "feat_dim": 40,
                "frame_length": 25,
                "frame_shift": 10,
                "dither": 0,
                "apply_cmvn": True,
                "delta_order": 2,
                "delta_window_size": 2,
            },
            "text": {
                "mode": "subword",
                "vocab_file": "spanish_subword.model",
            },
        },
        "hparas": {
            "valid_step": 5000,
            "max_step": 1000001,
            "tf_start": 1.0,
            "tf_end": 1.0,
            "tf_step": 500000,
            "optimizer": "Adadelta",
            "lr": 1.0,
            "eps": 1e-08,
            "lr_scheduler": "fixed",
            "curriculum": 0,
        },
        "model": {
            "ctc_weight": 0.0,
            "encoder": {
                "prenet": "vgg",
                "module": "LSTM",
                "bidirection": True,
                "dim": [dim] * num_layers,
                "dropout": [0]*num_layers,
                "layer_norm": [False]*num_layers,
                "proj": [True]*num_layers,
                "sample_rate": [1]*num_layers,
                "sample_style": "drop",
            },
            "attention": {
                "mode": "loc",
                "dim": 300,
                "num_head": 1,
                "v_proj": False,
                "temperature": 0.5,
                "loc_kernel_size": 100,
                "loc_kernel_num": 10,
            },
            "decoder": {"module": "LSTM", "dim": dim, "layer": 1, "dropout": 0},
        },
    }

    if not all([os.path.isfile(f) for f in TRAIN_EVAL_DATASET_FILES]):
        data = list(spanish_corpus(config['data']['corpus']['path']).items())
        idx = list(range(len(data)))
        random.shuffle(idx)
        train_eval_split_idx = round(len(data) * 0.9)
        train_file,eval_file = TRAIN_EVAL_DATASET_FILES
        data_io.write_jsonl(train_file, [data[k] for k in idx[:train_eval_split_idx]])
        data_io.write_jsonl(eval_file, [data[k] for k in idx[train_eval_split_idx:]])

    params = argparse.Namespace(**params)
    solver = Solver(config, params, mode="train")
    solver.load_data()
    solver.set_model()
    solver.exec()

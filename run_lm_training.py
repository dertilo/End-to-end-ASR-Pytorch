import argparse

from bin.train_lm import Solver

if __name__ == "__main__":
    params = {
        "config": "config/spanish/lm.yaml",
        "name": None,
        "logdir": "log/",
        "ckpdir": "ckpt/",
        "outdir": "result/",
        "load": None,
        "seed": 0,
        "cudnn_ctc": False,
        "njobs": 6,
        "cpu": True,
        "no_pin": False,
        "test": False,
        "no_msg": False,
        "lm": True,
        "amp": False,
        "reserve_gpu": 0,
        "jit": False,
        "gpu": False,
        "pin_memory": True,
        "verbose": True,
    }

    config = {
        "data": {
            "corpus": {
                "name": "spanish",
                "path": "/home/tilo/data/asr_data",
                "train_split": ["es_co_female"],
                "dev_split": ["es_co_female"],
                "bucketing": True,
                "batch_size": 32,
            },
            "text": {
                "mode": "subword",
                "vocab_file": "tests/sample_data/subword-16k.model",
            },
        },
        "hparas": {
            "valid_step": 100,
            "max_step": 1000,
            "optimizer": "Adam",
            "lr": 0.0001,
            "eps": 1e-08,
            "lr_scheduler": "fixed",
        },
        "model": {
            "emb_tying": False,
            "emb_dim": 64,
            "module": "LSTM",
            "dim": 64,
            "n_layers": 2,
            "dropout": 0.5,
        },
    }
    # config = argparse.Namespace(**config)
    params = argparse.Namespace(**params)
    solver = Solver(config, params, mode='train')
    solver.load_data()
    solver.set_model()
    solver.exec()
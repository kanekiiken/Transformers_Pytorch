from pathlib import Path


def get_config():
    return {
        "batct_size": 8,
        "num_epochs": 20,
        "lr": 1e-4,
        "seq_len": 350,
        "d_model": 512,
        "n_head": 4,
        "d_ffn": 1024,
        "lang_src": "en",
        "lang_tgt": "gr",
        "model_folder": "checkpoints",
        "preload": None,
        "tokenizer_file": "tokenizer_{0}.json",
        "experiment_name": "runs/tmodel",
    }


def get_weights_file_path(config, _epoch: str):
    model_folder = config["model_folder"]
    model_basename = config["model_basename"]
    model_filename = f"{model_basename}_{_epoch}.pt"
    return str(Path(".") / model_folder / model_filename)

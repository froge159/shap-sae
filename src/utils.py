from sae_lens import HookedSAETransformer
import torch
import json
from datasets import load_dataset


def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HookedSAETransformer.from_pretrained_no_processing("gpt2", device=device)
    return model

def load_splits():
    ds = load_dataset("stanfordnlp/sst2", split="train")
    with open("data/three_way_split_indices.json") as f:
        indices = json.load(f)

    train_ds = ds.select(indices["train_indices"])
    val_ds   = ds.select(indices["val_indices"])
    shap_ds  = ds.select(indices["shap_indices"])

    return train_ds, val_ds, shap_ds
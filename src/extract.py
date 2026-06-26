from sae_lens import SAE, HookedSAETransformer
import torch
import json
from datasets import load_dataset
import numpy as np
from tqdm import tqdm
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TARGET_LAYERS = tuple(range(7, 12))
SAE_RELEASE = "gpt2-small-resid-post-v5-32k"


def main():
    # call create_splits() if first time running
    model = load_model()
    saes = load_saes()
    train_ds, val_ds, shap_ds = load_splits()
    save_split(model, saes, train_ds, "activations/probe_train")
    save_split(model, saes, val_ds, "activations/probe_val")
    save_split(model, saes, shap_ds, "activations/shap")
    print(
        f"Activations saved for layers {TARGET_LAYERS} under "
        "'activations/probe_train', 'activations/probe_val', and 'activations/shap'"
    )


def load_model():
    model = HookedSAETransformer.from_pretrained_no_processing("gpt2", device=device)
    return model


def load_saes():
    saes = {}
    for layer in TARGET_LAYERS:
        sae_id = f"blocks.{layer}.hook_resid_post"
        sae = SAE.from_pretrained(SAE_RELEASE, sae_id)
        sae.to(device)
        saes[layer] = sae
    return saes


def load_splits():
    ds = load_from_disk("data/sst2_train")
    with open("data/three_way_split_indices.json") as f:
        indices = json.load(f)

    train_ds = ds.select(indices["train_indices"])
    val_ds = ds.select(indices["val_indices"])
    shap_ds = ds.select(indices["shap_indices"])

    return train_ds, val_ds, shap_ds


def create_splits():
    ds = load_from_disk("data/sst2_train")
    ds_with_id = ds.add_column("original_index", range(len(ds)))
    first_split = ds_with_id.train_test_split(test_size=0.30, seed=42)

    train_ds = first_split["train"]
    temp_ds = first_split["test"]

    second_split = temp_ds.train_test_split(test_size=0.5, seed=42)

    val_ds = second_split["train"]
    shap_ds = second_split["test"]

    train_indices = list(train_ds["original_index"])
    val_indices = list(val_ds["original_index"])
    shap_indices = list(shap_ds["original_index"])

    indices_dict = {
        "train_indices": train_indices,
        "val_indices": val_indices,
        "shap_indices": shap_indices,
    }

    Path("data").mkdir(parents=True, exist_ok=True)
    with open("data/three_way_split_indices.json", "w") as f:
        json.dump(indices_dict, f, indent=4)

    ds.save_to_disk("data/sst2_train")

    print("Dataset three-way split complete!")
    print(f"Train samples (70%): {len(train_indices)}")
    print(f"Val samples   (15%): {len(val_indices)}")
    print(f"Test samples  (15%): {len(shap_indices)}")
    print("Indices saved successfully to 'data/three_way_split_indices.json'")


def sae_cache_key(layer: int) -> str:
    return f"blocks.{layer}.hook_resid_post.hook_sae_acts_post"


def extract_split_activations(model, saes, sentences, labels, batch_size=32):
    layers = sorted(saes.keys())
    per_layer_acts = {layer: [] for layer in layers}
    all_labels = []

    with torch.inference_mode():
        for i in tqdm(range(0, len(sentences), batch_size)):
            batch_sents = sentences[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            tokens = model.to_tokens(batch_sents)  # [B, T]
            _, cache = model.run_with_cache_with_saes(
                tokens, saes=[saes[layer] for layer in layers]
            )

            pad_id = model.tokenizer.pad_token_id
            if pad_id is None:
                pad_id = model.tokenizer.eos_token_id
            mask = tokens != pad_id
            last_idx = mask.sum(dim=1) - 1  # [B]
            batch_idx = torch.arange(tokens.shape[0], device=tokens.device)

            for layer in layers:
                sae_acts = cache[sae_cache_key(layer)]  # [B, T, F]
                sentence_acts = sae_acts[batch_idx, last_idx, :]  # [B, F]
                per_layer_acts[layer].append(sentence_acts.cpu().numpy())

            all_labels.append(np.array(batch_labels))

    labels_arr = np.concatenate(all_labels, axis=0)
    activations_by_layer = {
        layer: np.concatenate(per_layer_acts[layer], axis=0) for layer in layers
    }
    return activations_by_layer, labels_arr


def save_split(model, saes, dataset, out_dir):
    out_dir = Path(out_dir)
    acts_by_layer, labels = extract_split_activations(
        model,
        saes,
        dataset["sentence"],
        dataset["label"],
        batch_size=32,
    )
    for layer, acts in acts_by_layer.items():
        layer_dir = out_dir / f"layer_{layer}"
        layer_dir.mkdir(parents=True, exist_ok=True)
        np.save(layer_dir / "activations.npy", acts)
        np.save(layer_dir / "labels.npy", labels)
        print(layer_dir, acts.shape, labels.shape)





if __name__ == "__main__":
    main()

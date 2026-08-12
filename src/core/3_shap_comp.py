import sys
from pathlib import Path

import numpy as np
import joblib

_SRC = Path(__file__).resolve().parents[1]
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from utils import (
    checkpoint_path,
    get_shap_feature_mask,
    load_activations,
    output_path,
)


def main(
    checkpoint_dir: Path | None = None,
    layer: int = 7,
    out_dir: Path | None = None,
):
    """
    Compute and save the SHAP-filtered feature mask (`utils.get_shap_feature_mask`)
    for downstream consumers.

    This used to also run KernelSHAP and save its Φ, but nothing downstream reads
    those outputs — global_linear/7_shap_recompute.py's exact LinearSHAP superseded
    it — so that computation was removed. `outputs/3_shap/shap_feature_indices.npy`
    is still load-bearing: 11_mlp_probe.py, 9_steer.py, 9.5_tuning.py,
    10_simplicity_check.py, group_steering.py, and local/9_local_steer.py all load
    it directly by this path.
    """
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else checkpoint_path()
    out_dir = Path(out_dir) if out_dir else output_path("3_shap")

    probe = joblib.load(checkpoint_dir / f"probe_layer_{layer}.joblib")
    probe.verbose = 0

    val_activations = load_activations("val", layer, mmap=True)
    feature_indices = get_shap_feature_mask(probe, val_activations)

    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "shap_feature_indices.npy", feature_indices)
    print(f"\nWrote {len(feature_indices)} filtered feature indices -> {out_dir}/shap_feature_indices.npy")


if __name__ == "__main__":
    main()

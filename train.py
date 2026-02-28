# train.py
import time
from pathlib import Path
from contextlib import redirect_stdout

import numpy as np

RESULTS_DIR = Path(__file__).parent / "results.txt"

from src.data_prep import (
    load_data,
    filter_amount,
    undersample_majority,
    split_X_y,
)
from src.autoencoder import (
    scale_features_minmax,
    split_normal_fraud,
    train_autoencoder,
    build_encoder,
    latent_representations,
    save_all_val_losses,
)
from src.tsne_visualization import tsne_embeddings, plot_tsne
from src.models_eval import (
    eval_logreg,
    eval_decision_tree,
    eval_xgboost,
    eval_random_forest,
)


def log(msg: str):
    text = str(msg)
    print(msg, flush=True)
    with RESULTS_DIR.open("a", encoding="utf-8") as f:
        f.write(text + "\n")

def main():
    if RESULTS_DIR.exists():
        RESULTS_DIR.unlink()

    start = time.perf_counter()
    log("Starting pipeline...")

    data_path = Path("data/creditcard.csv")

    # 1. Load and filter by Amount
    df_raw = load_data(data_path)
    log(f"Loaded  {df_raw.shape[0]} rows")

    df = filter_amount(df_raw, max_amount=8000.0)
    log(f"After Amount filter: {df.shape[0]} rows")

    # 2. Build undersampled datasets (1:10, 1:20, 1:50)
    sampled_dfs = undersample_majority(df, ratios=(10, 20, 50))
    log("Created undersampled datasets: " + ", ".join(sampled_dfs.keys()))

    # 3. Split to X, y
    X_sets = {}
    y_sets = {}
    for key, sdf in sampled_dfs.items():
        X, y = split_X_y(sdf, target_col="Class")
        X_sets[key] = X
        y_sets[key] = y
    log("Split sampled datasets into X and y")

    # 4. (Optional) TSNE on original high-dim data
    log("Computing t-SNE on original feature space...")
    highdim_tsne = tsne_embeddings(X_sets)
    for key, emb in highdim_tsne.items():
        print("saving tsne representation for " + key) # debug
        plot_tsne(
            emb,
            y_sets[key],
            f"t-SNE plot ({key})",
            subdir = "tsne_representation",
            filename = f"tsne_representation_{key}.png",
        )
    log("Finished t-SNE on original features")

    # 5. Scale + autoencoders per ratio
    repr_X = {}
    repr_y = {}

    histories = {}

    for key in ("1_to_10", "1_to_20", "1_to_50"):
        log(f"Scaling and training autoencoder for {key}...")
        X = X_sets[key]
        y = y_sets[key]

        X_scaled = scale_features_minmax(X)
        X_norm, X_fraud = split_normal_fraud(X_scaled, y)

        autoencoder, history = train_autoencoder(X_norm)
        encoder = build_encoder(autoencoder)

        X_repr, y_repr, _ = latent_representations(encoder, X_norm, X_fraud)
        repr_X[key] = X_repr
        repr_y[key] = y_repr
        histories[key] = history
        log(f"Finished autoencoder + latent representation for {key}")

    save_all_val_losses(
        histories["1_to_10"],
        histories["1_to_20"],
        histories["1_to_50"],
    )

    # 6. TSNE on latent representations
    log("Computing t-SNE on latent representations...")
    latent_tsne = tsne_embeddings(repr_X, n_components=2, random_state=24)
    for key in ("1_to_10", "1_to_20", "1_to_50"):
        plot_tsne(
            latent_tsne[key],
            repr_y[key],
            f"t-SNE (latent, {key})",
            subdir = "tsne_latent_representation",
            filename = f"tsne_latent_representation_{key}.png",
        )
    log("Finished t-SNE on latent space")

    with RESULTS_DIR.open("a", encoding="utf-8") as f:
        with redirect_stdout(f):
            # 7. Evaluate models on latent representations
            log("Evaluating models on latent representations...")

            for key, label in [("1_to_10", "1:10"), ("1_to_20", "1:20"), ("1_to_50", "1:50")]:
                Xr = repr_X[key]
                yr = repr_y[key]
                eval_logreg(Xr, yr, label)
                eval_decision_tree(Xr, yr, label)
                eval_xgboost(Xr, yr, label)
                eval_random_forest(Xr, yr, label)

            end = time.perf_counter()
            log(f"Total runtime: {end - start:.2f} seconds")

if __name__ == "__main__":
    main()

from typing import Dict

import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from src.plot_utils import plot_path


def tsne_embeddings(
    datasets: Dict[str, np.ndarray],
    n_components: int = 2,
    random_state: int = 24,
) -> Dict[str, np.ndarray]:
    """Compute TSNE embeddings for multiple datasets."""
    out: Dict[str, np.ndarray] = {}
    for name, X in datasets.items():
        emb = TSNE(n_components=n_components, random_state=random_state).fit_transform(X)
        out[name] = emb
    return out


def plot_tsne(embedding, y, title: str, subdir: str, filename: str):
    colour_map = {0: "green", 1: "red"}
    name_map = {0: "Non-fraud", 1: "Fraud"}

    plt.figure()
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            x=embedding[y == cl, 0],
            y=embedding[y == cl, 1],
            c=colour_map[idx],
            label=name_map[cl],
        )
    plt.xlabel("X in t-SNE space")
    plt.ylabel("Y in t-SNE space")
    plt.legend(loc="upper right")
    plt.title(title)

    out_path = plot_path(subdir, filename)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
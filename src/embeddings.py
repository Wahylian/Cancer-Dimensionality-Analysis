"""Phase 3: linear and nonlinear embedding wrappers (PCA, t-SNE, UMAP)."""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

from src import config


def run_pca(X: np.ndarray, n_components: int) -> dict:
    """Project X onto its top `n_components` principal components.

    Uses the full (exact) SVD solver: with n=801 samples this is cheap and,
    unlike the randomized solver, gives an exactly reproducible embedding
    independent of any internal random state.

    Returns
    -------
    dict with keys 'embedding' (n_samples x n_components) and
    'explained_variance_ratio' (n_components,).
    """
    pca = PCA(n_components=n_components, svd_solver="full", random_state=config.RANDOM_SEED)
    embedding = pca.fit_transform(X)
    return {"embedding": embedding, "explained_variance_ratio": pca.explained_variance_ratio_}


def _pca_preprocess(X: np.ndarray, seed: int) -> np.ndarray:
    """Reduce X to config.EMBEDDING_PCA_PREPROCESS_DIM dims before t-SNE/UMAP.

    Standard practice for nonlinear embeddings on very high-dimensional input:
    denoises minor components and turns an O(n^2 * d) neighbor search into
    O(n^2 * 50), without discarding structure (Phase 2 showed 90% of linear
    variance is captured well below 50 components on this data).
    """
    n_components = min(config.EMBEDDING_PCA_PREPROCESS_DIM, X.shape[1])
    return PCA(n_components=n_components, svd_solver="full", random_state=seed).fit_transform(X)


def run_tsne(X: np.ndarray, perplexities: list, seed: int) -> dict:
    """Return {perplexity: 2D embedding} for each value in `perplexities`."""
    X_reduced = _pca_preprocess(X, seed)
    embeddings = {}
    for perplexity in perplexities:
        tsne = TSNE(
            n_components=2,
            perplexity=perplexity,
            init="pca",
            learning_rate="auto",
            random_state=seed,
        )
        embeddings[perplexity] = tsne.fit_transform(X_reduced)
    return embeddings


def run_umap(X: np.ndarray, n_neighbors_values: list, seed: int) -> dict:
    """Return {n_neighbors: 2D embedding} for each value in `n_neighbors_values`."""
    X_reduced = _pca_preprocess(X, seed)
    embeddings = {}
    for n_neighbors in n_neighbors_values:
        reducer = UMAP(n_neighbors=n_neighbors, n_components=2, random_state=seed)
        embeddings[n_neighbors] = reducer.fit_transform(X_reduced)
    return embeddings

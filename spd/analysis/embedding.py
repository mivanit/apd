from typing import Literal, Sequence, Union, Any
from dataclasses import dataclass
import numpy as np
from sklearn.base import TransformerMixin
from sklearn.manifold import TSNE, Isomap
from umap import UMAP
from jaxtyping import Float
import matplotlib.pyplot as plt
from muutils.collect_warnings import CollateWarnings

# TODO: this is ugly af
NDArray = np.ndarray[Any, Any]

ReduceMethod = Literal["umap", "isomap", "tsne"]
ParamValue = Union[int, float, str]

@dataclass
class EmbeddingResult:
    method: ReduceMethod
    param_name: str
    param_values: list[ParamValue]
    embeddings: dict[str, Float[NDArray, "n_components n_features"]]


def get_embedding_model(
    method: ReduceMethod,
    kwargs: dict[str, Any],
    n_components: int = 2,
    random_state: int|None = None,
) -> TransformerMixin:
    """Return configured embedding model."""
    if method == "umap":
        return UMAP(n_components=n_components, metric="precomputed", random_state=random_state, **kwargs) # type: ignore
    elif method == "isomap":
        return Isomap(n_components=n_components, metric="precomputed", **kwargs)
    elif method == "tsne":
        return TSNE( # type: ignore
            n_components=n_components,
            metric="precomputed",
            random_state=random_state,
            init="random",
            learning_rate="auto",
            **kwargs
        )
    raise ValueError(f"Unsupported method: {method}")


def compute_embedding_sweep(
    dist: NDArray,
    method: ReduceMethod,
    param_grid: list[dict[str, Any]],
    n_components: int = 2,
    random_state: int|None = None,
) -> dict[str, NDArray]:
    """Compute embeddings for a grid of parameter dicts."""
    embeddings: dict[str, NDArray] = {}
    for kwargs in param_grid:
        key: str = "_".join(f"{k}={v}" for k, v in kwargs.items())
        model: TransformerMixin = get_embedding_model(method, kwargs, n_components, random_state)
        embeddings[key] = model.fit_transform(dist)
    return embeddings


def sweep_embedding_param(
    dist: NDArray,
    method: ReduceMethod,
    param_name: str,
    param_values: list[ParamValue],
    n_components: int = 2,
    random_state: int|None = None,
) -> EmbeddingResult:
    """Sweep a single hyperparameter and return results."""
    with CollateWarnings(fmt = "({count}x) {filename}:{lineno}\n  {category}: {message}"):
        param_grid: list[dict[str, Any]] = [{param_name: val} for val in param_values]
        embeddings: dict[str, NDArray] = compute_embedding_sweep(dist, method, param_grid, n_components, random_state)

    return EmbeddingResult(method=method, param_name=param_name, param_values=param_values, embeddings=embeddings)



def plot_embedding_result(
    result: EmbeddingResult,
    figsize: tuple[int, int]|None = None,
) -> None:
    """Plot all 2D embeddings in a row."""
    n: int = len(result.param_values)
    figsize = figsize or (4 * n, 4)
    axes: Sequence[plt.Axes]
    fig, axes = plt.subplots(1, n, figsize=figsize) # type: ignore
    for ax, param_val in zip(axes, result.param_values, strict=True):
        key: str = f"{result.param_name}={param_val}"
        emb: NDArray = result.embeddings[key]
        ax.scatter(emb[:, 0], emb[:, 1])
        ax.set_title(f"{result.method.upper()} {key}")
        ax.grid(True)

    plt.tight_layout()
    plt.show()
    
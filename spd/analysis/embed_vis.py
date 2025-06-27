from dataclasses import dataclass
from itertools import product
from typing import Any, Literal

import numpy as np
import pandas as pd
from jaxtyping import Float
from muutils.collect_warnings import CollateWarnings
from sklearn.base import TransformerMixin

from spd.analysis.embedding import (
    ClusteringMethod,
    NDArray,
    ReduceMethod,
    get_clustering_model,
    get_comp_dist_mat,
    get_embedding_model,
)

# Import your existing types and functions
from spd.analysis.grouping import CoactivationResultsGroup


@dataclass
class AnalysisConfig:
    """Configuration for the comprehensive analysis."""

    # Embedding methods and their hyperparameters
    embedding_configs: dict[ReduceMethod, dict[str, list[Any]]] = None

    # Clustering methods and their hyperparameters
    clustering_configs: dict[ClusteringMethod, dict[str, list[Any]]] = None

    # Hierarchical clustering config
    hclust_threshold: float = 0.8
    hclust_linkage: Literal["single", "complete", "average", "ward"] = "average"
    hclust_min_alive_counts: int = 500

    # Distance matrix config
    dist_epsilon: float = 1.0
    dist_normalize: bool = True

    # Random state for reproducibility
    random_state: int | None = 42

    # Number of embedding components
    n_components: int = 3

    def __post_init__(self) -> None:
        if self.embedding_configs is None:
            self.embedding_configs = {
                "umap": {"n_neighbors": [2, 4, 8, 16, 32, 64, 128]},
                "isomap": {"n_neighbors": [2, 4, 8, 16, 32, 64, 128]},
                "tsne": {"perplexity": [5, 10, 20, 30, 50]},
            }

        if self.clustering_configs is None:
            self.clustering_configs = {
                "kmeans": {"n_clusters": [3, 5, 8, 10, 15, 20]},
                "agglomerative": {"n_clusters": [3, 5, 8, 10, 15, 20]},
                "spectral": {"n_clusters": [3, 5, 8, 10, 15, 20]},
                "dbscan": {"eps": [0.05, 0.1, 0.2, 0.3, 0.5]},
                "optics": {"min_samples": [3, 5, 10, 20]},
            }


@dataclass
class DataFrameMetadata:
    """Metadata describing the structure of the analysis DataFrame."""

    n_features: int
    n_alive_features: int
    embedding_methods: list[str]
    clustering_methods: list[str]
    column_groups: dict[str, list[str]]
    hyperparameter_configs: dict[str, dict[str, Any]]

    def describe(self) -> str:
        """Human-readable description of the DataFrame structure."""
        desc = f"Analysis df with {self.n_features} features ({self.n_alive_features} alive)\n\n"

        desc += "Column Groups:\n"
        for group_name, cols in self.column_groups.items():
            desc += f"  {group_name}: {len(cols)} columns\n"
            if len(cols) <= 10:
                desc += f"    {cols}\n"
            else:
                desc += f"    {cols[:5]} ... {cols[-2:]}\n"

        desc += f"\nEmbedding methods: {', '.join(self.embedding_methods)}\n"
        desc += f"Clustering methods: {', '.join(self.clustering_methods)}\n"

        return desc


def _format_hyperparam_str(params: dict[str, Any]) -> str:
    """Format hyperparameters into a standardized string representation."""
    return ".".join(f"{k}-{v}" for k, v in sorted(params.items()))


def coactivation_analysis(
    group: CoactivationResultsGroup,
    config: AnalysisConfig,
    verbose: bool = True,
) -> tuple[pd.DataFrame, DataFrameMetadata]:
    """
    analysis of a coactivation group, returning a DataFrame with embeddings and clusterings.

    # Parameters:
     - `group : CoactivationResultsGroup`
        Single group from coactivation results (e.g., `coactivations_tms["group_0"]`)
     - `config : AnalysisConfig | None`
        Configuration for analysis parameters (defaults to `AnalysisConfig()`)
     - `verbose : bool`
        Whether to print progress information (defaults to `True`)

    # Returns:
     - `pd.DataFrame`
        Comprehensive analysis results with columns:
        - `feat.alive`: boolean indicating if feature is alive
        - `feat.activation_freq`: activation frequency
        - `feat.module`: module name for each feature
        - `feat.class.hclust`: hierarchical clustering labels
        - `embed.{method}.{hparams}.ax.{i}`: embedding coordinates
        - `feat.class.{embed_method}.{embed_hparams}.{cluster_method}.{cluster_hparams}`:
            cluster labels
     - `DataFrameMetadata`
        Metadata describing the DataFrame structure
    """

    if verbose:
        print("Starting coactivation analysis...")

    # Get distance matrix
    dist_mat: Float[NDArray, "n n"] = get_comp_dist_mat(
        group,
        verbose=verbose,
        plots=False,  # Suppress plots for cleaner output
        epsilon=config.dist_epsilon,
        normalize_dist=config.dist_normalize,
    )

    n_features: int = len(group["labels"])

    # Initialize DataFrame with basic feature info
    df_data: dict[str, Any] = {
        "feat.alive": group["is_alive"].cpu().numpy(),
        "feat.activation_freq": group["active_freq"].cpu().numpy(),
        "feat.module": group["labels"],
    }

    # Track metadata
    column_groups: dict[str, list[str]] = {
        "feat": ["feat.alive", "feat.activation_freq", "feat.module"],
        "embed": [],
        "feat.class": [],
    }

    # Hierarchical clustering (on alive features only)
    if verbose:
        print("Computing hierarchical clustering...")

    from spd.analysis.grouping import coactivation_hierarchical_clustering

    hclust_result = coactivation_hierarchical_clustering(
        group,
        threshold=config.hclust_threshold,
        linkage_method=config.hclust_linkage,
        min_alive_counts=config.hclust_min_alive_counts,
        plot=False,  # Suppress plots
    )

    # Add hierarchical clustering results
    df_data["feat.class.hclust"] = hclust_result["clusters_nomask"]
    column_groups["feat.class"].append("feat.class.hclust")

    # Get alive mask for subsequent analyses
    alive_mask: NDArray = hclust_result["alive_mask"]
    alive_dist_mat: Float[NDArray, "n_alive n_alive"] = dist_mat[alive_mask][:, alive_mask]

    if verbose:
        print(f"Working with {alive_mask.sum()} alive features out of {n_features} total")

    # Embedding analysis
    embedding_results: dict[str, NDArray] = {}

    for emb_method, param_configs in config.embedding_configs.items():
        if verbose:
            print(f"Computing {emb_method} embeddings...")

        # Generate all parameter combinations
        param_names = list(param_configs.keys())
        param_values = list(param_configs.values())

        for param_combo in product(*param_values):
            params = dict(zip(param_names, param_combo, strict=False))
            param_str = _format_hyperparam_str(params)

            try:
                with CollateWarnings():
                    emb_model: TransformerMixin = get_embedding_model(
                        emb_method,
                        params,
                        n_components=config.n_components,
                        random_state=config.random_state,
                    )
                    emb_coords: NDArray = emb_model.fit_transform(alive_dist_mat)

                # Store embedding coordinates
                emb_key = f"{emb_method}.{param_str}"
                embedding_results[emb_key] = emb_coords

                # Add to DataFrame (padded with NaN for non-alive features)
                for axis in range(config.n_components):
                    col_name = f"embed.{emb_method}.{param_str}.ax.{axis}"
                    full_coords = np.full(n_features, np.nan)
                    full_coords[alive_mask] = emb_coords[:, axis]
                    df_data[col_name] = full_coords
                    column_groups["embed"].append(col_name)

            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to compute {emb_method} with {params}: {e}")
                continue

    # Clustering analysis (on embeddings)
    if verbose:
        print("Computing clustering on embeddings...")

    for emb_key, emb_coords in embedding_results.items():
        emb_method, emb_param_str = emb_key.split(".", 1)

        for clust_method, param_configs in config.clustering_configs.items():
            # Generate all parameter combinations
            param_names = list(param_configs.keys())
            param_values = list(param_configs.values())

            for param_combo in product(*param_values):
                params = dict(zip(param_names, param_combo, strict=False))
                param_str = _format_hyperparam_str(params)

                try:
                    clust_model = get_clustering_model(
                        clust_method, params, random_state=config.random_state
                    )
                    cluster_labels: NDArray = clust_model.fit_predict(emb_coords)

                    # Add to DataFrame (padded with -1 for non-alive features)
                    col_name = f"feat.class.{emb_method}.{emb_param_str}.{clust_method}.{param_str}"
                    full_labels = np.full(n_features, -1, dtype=int)
                    full_labels[alive_mask] = cluster_labels
                    df_data[col_name] = full_labels
                    column_groups["feat.class"].append(col_name)

                except Exception as e:
                    if verbose:
                        print(f"  Warning: Failed {clust_method} on {emb_key} with {params}: {e}")
                    continue

    # Create DataFrame
    df = pd.DataFrame(df_data)

    # Create metadata
    metadata = DataFrameMetadata(
        n_features=n_features,
        n_alive_features=int(alive_mask.sum()),
        embedding_methods=list(config.embedding_configs.keys()),
        clustering_methods=list(config.clustering_configs.keys()),
        column_groups=column_groups,
        hyperparameter_configs={
            "embedding": config.embedding_configs,
            "clustering": config.clustering_configs,
            "hclust": {
                "threshold": config.hclust_threshold,
                "linkage": config.hclust_linkage,
                "min_alive_counts": config.hclust_min_alive_counts,
            },
        },
    )

    if verbose:
        print(f"Analysis complete! DataFrame shape: {df.shape}")
        print(f"Column groups: {[f'{k}({len(v)})' for k, v in column_groups.items()]}")

    return df, metadata

from langchain_core.tools import tool
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import OneHotEncoder
import umap
import matplotlib.pyplot as plt
import seaborn as sns

def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: str = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: int = None
    ) -> dict:
        """Reduce dimensionality of the global document-term matrix using PCA, t-SNE, or UMAP, and generate a scatter plot.

        Args:
            method (str): Dimensionality reduction method ("pca", "tsne", or "umap", default "pca").
            perplexity (float): Perplexity parameter for t-SNE (default 30.0).
            n_neighbors (int): Number of neighbors parameter for UMAP (default 15).
            random_state (int): Random seed for reproducibility (default None).

        Returns:
            dict: Summary including reduced coordinates and explained variance ratio (if PCA).
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix built. Please build the DTM first."}

        X = session.feature_matrix.toarray()
        
        # Determine labels/categories for coloring
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        # Perform dimensionality reduction
        method_lower = method.lower()
        explained_variance_ratio = None

        if method_lower == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = reducer.explained_variance_ratio_.tolist()
        elif method_lower == "tsne":
            reducer = TSNE(n_components=2, perplexity=perplexity, random_state=random_state)
            coords = reducer.fit_transform(X)
        elif method_lower == "umap":
            reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
        else:
            return {"error": f"Unknown reduction method: {method}"}

        # -------------------------------------------------------------------------
        # PLOTTING REQUIREMENT:
        # reduce_dimensions_tool must produce a real plot and set it on session.pending_figure
        # -------------------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(8, 6))
        
        df_plot = pd.DataFrame({
            "Dim1": coords[:, 0],
            "Dim2": coords[:, 1]
        })
        if labels is not None:
            df_plot["category"] = labels
            sns.scatterplot(
                data=df_plot,
                x="Dim1",
                y="Dim2",
                hue="category",
                palette="Set1",
                ax=ax,
                s=100
            )
            ax.legend(title="Category")
        else:
            sns.scatterplot(
                data=df_plot,
                x="Dim1",
                y="Dim2",
                ax=ax,
                s=100
            )
            
        ax.set_title(f"Dimensionality Reduction ({method.upper()})")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()

        session.pending_figure = fig

        # Store result and artifacts
        result_id = session.next_result_id("reduce_dimensions")
        summary = {
            "result_id": result_id,
            "method": method_lower,
            "n_samples": int(X.shape[0]),
            "n_components": 2,
            "explained_variance_ratio": explained_variance_ratio,
            "coordinates": coords.tolist()
        }

        session.artifacts[f"reduced_{method_lower}"] = coords
        session.store_result("reduce_dimensions_tool", {"method": method, "perplexity": perplexity, "n_neighbors": n_neighbors, "random_state": random_state}, summary)

        return summary

    @tool
    def binarize_labels_tool() -> dict:
        """One-hot encode category labels from session using sklearn OneHotEncoder.

        Returns:
            dict: Summary containing classes, matrix shape, and binarized matrix rows.
        """
        labels = session.labels
        if labels is None and session.dataframe is not None and "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values

        if labels is None or len(labels) == 0:
            return {"error": "No labels available in session. Please load dataset or set labels first."}

        # Reshape to column vector for OneHotEncoder
        y = np.array(labels).reshape(-1, 1)
        encoder = OneHotEncoder(sparse_output=False, dtype=int)
        binary_matrix = encoder.fit_transform(y)
        classes = [str(c) for c in encoder.categories_[0]]

        session.artifacts["binarized_labels"] = binary_matrix

        result_id = session.next_result_id("binarize_labels")
        summary = {
            "result_id": result_id,
            "classes": classes,
            "shape": list(binary_matrix.shape),
            "binary_matrix": binary_matrix.tolist()
        }

        session.store_result("binarize_labels_tool", {}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]

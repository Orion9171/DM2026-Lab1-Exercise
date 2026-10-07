from langchain_core.tools import tool
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc_index_1: int,
        doc_index_2: int
    ) -> dict:
        """Compute the cosine similarity between two document rows in session.feature_matrix.

        Args:
            doc_index_1 (int): Index of the first document.
            doc_index_2 (int): Index of the second document.

        Returns:
            dict: Summary containing document indices and their cosine similarity score.
        """
        if session.feature_matrix is None:
            return {"error": "No feature matrix built. Please build the DTM first."}

        n_docs = session.feature_matrix.shape[0]
        if doc_index_1 < 0 or doc_index_1 >= n_docs:
            return {"error": f"Document index 1 ({doc_index_1}) is out of bounds for {n_docs} documents."}
        if doc_index_2 < 0 or doc_index_2 >= n_docs:
            return {"error": f"Document index 2 ({doc_index_2}) is out of bounds for {n_docs} documents."}

        # Extract rows using 2D slicing
        vec1 = session.feature_matrix[doc_index_1:doc_index_1+1, :]
        vec2 = session.feature_matrix[doc_index_2:doc_index_2+1, :]

        sim = float(cosine_similarity(vec1, vec2)[0, 0])

        result_id = session.next_result_id("cosine_similarity")
        summary = {
            "result_id": result_id,
            "doc_index_1": doc_index_1,
            "doc_index_2": doc_index_2,
            "cosine_similarity": sim
        }

        session.store_result("cosine_similarity_tool", {"doc_index_1": doc_index_1, "doc_index_2": doc_index_2}, summary)

        return summary

    @tool
    def feature_correlation_matrix_tool() -> dict:
        """Compute feature-vs-feature Pearson correlation matrix over the top-20 terms by variance from the global DTM, generate a heatmap, and set session.pending_figure.

        Returns:
            dict: Summary containing feature names and correlation matrix.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No feature matrix built. Please build the DTM first."}

        X = session.feature_matrix.toarray()
        feature_names = np.array(session.feature_names)

        if X.shape[1] == 0:
            return {"error": "Feature matrix has no columns."}

        # Compute variance for each term (column)
        variances = np.var(X, axis=0)
        
        # Sort by variance descending and take top 20 (or all if fewer than 20)
        top_k = min(20, len(feature_names))
        sorted_indices = np.argsort(variances)[::-1][:top_k]

        X_top = X[:, sorted_indices]
        top_features = feature_names[sorted_indices].tolist()

        # Compute Pearson correlation feature-to-feature (rowvar=False means columns are variables)
        corr_matrix = np.corrcoef(X_top, rowvar=False)
        if corr_matrix.ndim == 0:
            corr_matrix = np.array([[1.0]])

        # Build heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(
            corr_matrix,
            annot=True,
            fmt=".4f",
            xticklabels=top_features,
            yticklabels=top_features,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            ax=ax
        )
        ax.set_title("Feature-vs-Feature Correlation Matrix (Top Terms by Variance)")
        plt.tight_layout()

        session.pending_figure = fig

        result_id = session.next_result_id("feature_correlation")
        summary = {
            "result_id": result_id,
            "feature_names": top_features,
            "correlation_matrix": corr_matrix.tolist()
        }

        session.store_result("feature_correlation_matrix_tool", {}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]

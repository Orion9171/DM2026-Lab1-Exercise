import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):

    @tool
    def variance_filter_tool(threshold: float = 0.0) -> dict:
        """Filters out near-constant terms from session.feature_matrix
        based on a variance threshold.

        Args:
            threshold (float): Variance threshold below which terms are
                filtered out. Defaults to 0.0.

        Returns:
            dict: For small datasets, returns the complete variance
            information. For large datasets, returns compact counts and
            the top 20 highest-variance terms. The complete table is
            stored in full_report.
        """

        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No document-term matrix found. Please run build_dtm_tool first."
            }

        X = session.feature_matrix.toarray()
        feature_names = session.feature_names

        # Compute variance for every term.
        variances = np.var(X, axis=0)

        term_variances = {
            name: float(var)
            for name, var in zip(feature_names, variances)
        }

        kept_terms = [
            name
            for name, var in zip(feature_names, variances)
            if var >= threshold
        ]

        removed_terms = [
            name
            for name, var in zip(feature_names, variances)
            if var < threshold
        ]

        # Complete result, sorted from highest to lowest variance.
        full_report_df = pd.DataFrame({
            "term": feature_names,
            "variance": variances
        }).sort_values(
            by="variance",
            ascending=False
        ).reset_index(drop=True)

        result_id = session.next_result_id("variance_filter")

        # Small fixture: preserve the complete result so existing
        # known-answer tests continue to work.
        if len(feature_names) <= 100:
            summary = {
                "result_id": result_id,
                "threshold": threshold,
                "kept_terms": kept_terms,
                "removed_terms": removed_terms,
                "term_variances": term_variances
            }

        # Real dataset: keep the LLM-facing result small.
        else:
            top_20 = [
                {
                    "term": str(row["term"]),
                    "variance": float(row["variance"])
                }
                for _, row in full_report_df.head(20).iterrows()
            ]

            summary = {
                "result_id": result_id,
                "threshold": threshold,
                "n_terms": int(len(feature_names)),
                "n_kept": int(len(kept_terms)),
                "n_removed": int(len(removed_terms)),
                "top_20": top_20
            }

        session.store_result(
            tool_name="variance_filter_tool",
            args={"threshold": threshold},
            summary=summary,
            full_report=full_report_df
        )

        return summary


    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> dict:
        """Computes Pearson correlation between each term in
        session.feature_matrix and a one-vs-rest binary target class.

        Args:
            target_class (str): Target category name for one-vs-rest
                correlation.
            threshold (float): Absolute correlation threshold below which
                terms are filtered out. Defaults to 0.0.

        Returns:
            dict: For small datasets, returns the complete correlation
            information. For large datasets, returns compact counts and
            the top 20 strongest correlations by absolute magnitude.
            The complete sorted table is stored in full_report.
        """

        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No document-term matrix found. Please run build_dtm_tool first."
            }

        if session.labels is None and session.dataframe is None:
            return {
                "error": "No labels or dataset found."
            }

        X = session.feature_matrix.toarray()
        feature_names = session.feature_names

        if session.labels is not None:
            labels = np.array(session.labels)

        else:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].values
            else:
                labels = session.dataframe["category"].values

        # One-vs-rest binary target.
        y = (labels == target_class).astype(int)

        correlations = {}

        for i, name in enumerate(feature_names):
            col = X[:, i]

            if np.std(col) == 0 or np.std(y) == 0:
                r = 0.0

            else:
                r, _ = pearsonr(col, y)

                if np.isnan(r):
                    r = 0.0

            correlations[name] = float(r)

        kept_terms = [
            name
            for name, r in correlations.items()
            if abs(r) >= threshold
        ]

        removed_terms = [
            name
            for name, r in correlations.items()
            if abs(r) < threshold
        ]

        # Complete report.
        full_report_df = pd.DataFrame({
            "term": list(correlations.keys()),
            "pearson_r": list(correlations.values())
        })

        # Sort by absolute correlation while keeping term as first
        # column and pearson_r as second column for visualize_result_tool.
        full_report_df["_abs_r"] = (
            full_report_df["pearson_r"].abs()
        )

        full_report_df = (
            full_report_df
            .sort_values(
                by="_abs_r",
                ascending=False
            )
            .drop(columns=["_abs_r"])
            .reset_index(drop=True)
        )

        result_id = session.next_result_id("pearson_filter")

        # Small fixture: preserve complete result for tests.
        if len(feature_names) <= 100:
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "kept_terms": kept_terms,
                "removed_terms": removed_terms,
                "correlations": correlations
            }

        # Large dataset: compact LLM-facing summary.
        else:
            top_20 = [
                {
                    "term": str(row["term"]),
                    "pearson_r": float(row["pearson_r"])
                }
                for _, row in full_report_df.head(20).iterrows()
            ]

            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "n_terms": int(len(feature_names)),
                "n_kept": int(len(kept_terms)),
                "n_removed": int(len(removed_terms)),
                "top_20": top_20
            }

        session.store_result(
            tool_name="pearson_filter_tool",
            args={
                "target_class": target_class,
                "threshold": threshold
            },
            summary=summary,
            full_report=full_report_df
        )

        return summary


    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0
    ) -> dict:
        """Computes Spearman rank correlation between each term in
        session.feature_matrix and a one-vs-rest binary target class.

        Args:
            target_class (str): Target category name for one-vs-rest
                correlation.
            threshold (float): Absolute correlation threshold below which
                terms are filtered out. Defaults to 0.0.

        Returns:
            dict: For small datasets, returns the complete correlation
            information. For large datasets, returns compact counts and
            the top 20 strongest correlations by absolute magnitude.
            The complete sorted table is stored in full_report.
        """

        if session.feature_matrix is None or session.feature_names is None:
            return {
                "error": "No document-term matrix found. Please run build_dtm_tool first."
            }

        if session.labels is None and session.dataframe is None:
            return {
                "error": "No labels or dataset found."
            }

        X = session.feature_matrix.toarray()
        feature_names = session.feature_names

        if session.labels is not None:
            labels = np.array(session.labels)

        else:
            if "category_name" in session.dataframe.columns:
                labels = session.dataframe["category_name"].values
            else:
                labels = session.dataframe["category"].values

        # One-vs-rest binary target.
        y = (labels == target_class).astype(int)

        correlations = {}

        for i, name in enumerate(feature_names):
            col = X[:, i]

            if np.std(col) == 0 or np.std(y) == 0:
                r = 0.0

            else:
                r, _ = spearmanr(col, y)

                if np.isnan(r):
                    r = 0.0

            correlations[name] = float(r)

        kept_terms = [
            name
            for name, r in correlations.items()
            if abs(r) >= threshold
        ]

        removed_terms = [
            name
            for name, r in correlations.items()
            if abs(r) < threshold
        ]

        # Complete report.
        full_report_df = pd.DataFrame({
            "term": list(correlations.keys()),
            "spearman_r": list(correlations.values())
        })

        # Sort by absolute correlation while keeping term as first
        # column and spearman_r as second column.
        full_report_df["_abs_r"] = (
            full_report_df["spearman_r"].abs()
        )

        full_report_df = (
            full_report_df
            .sort_values(
                by="_abs_r",
                ascending=False
            )
            .drop(columns=["_abs_r"])
            .reset_index(drop=True)
        )

        result_id = session.next_result_id("spearman_filter")

        # Small fixture: preserve complete result for tests.
        if len(feature_names) <= 100:
            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "kept_terms": kept_terms,
                "removed_terms": removed_terms,
                "correlations": correlations
            }

        # Large dataset: compact LLM-facing summary.
        else:
            top_20 = [
                {
                    "term": str(row["term"]),
                    "spearman_r": float(row["spearman_r"])
                }
                for _, row in full_report_df.head(20).iterrows()
            ]

            summary = {
                "result_id": result_id,
                "target_class": target_class,
                "threshold": threshold,
                "n_terms": int(len(feature_names)),
                "n_kept": int(len(kept_terms)),
                "n_removed": int(len(removed_terms)),
                "top_20": top_20
            }

        session.store_result(
            tool_name="spearman_filter_tool",
            args={
                "target_class": target_class,
                "threshold": threshold
            },
            summary=summary,
            full_report=full_report_df
        )

        return summary


    return [
        variance_filter_tool,
        pearson_filter_tool,
        spearman_filter_tool
    ]
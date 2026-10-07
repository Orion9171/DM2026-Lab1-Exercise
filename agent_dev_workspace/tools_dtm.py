import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def build_dtm_tool(ngram_range: list = [1, 1], max_features: int = None) -> dict:
        """Builds a document-term matrix from session.dataframe['text'] using CountVectorizer,
        populating session.feature_matrix, session.feature_names, and session.artifacts["count_vectorizer"].
        
        Args:
            ngram_range (list): N-gram range as a list of two ints [min_n, max_n]. Defaults to [1, 1].
            max_features (int, optional): Maximum vocabulary size. Defaults to None.
            
        Returns:
            dict: Summary containing vocabulary size, sparsity percentage, non-zero elements, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
            
        texts = session.dataframe["text"].fillna("").astype(str).tolist()
        ngram_tuple = tuple(ngram_range)
        
        vectorizer = CountVectorizer(ngram_range=ngram_tuple, max_features=max_features)
        X = vectorizer.fit_transform(texts)
        feature_names = vectorizer.get_feature_names_out().tolist()
        
        session.feature_matrix = X
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer
        
        if "category_name" in session.dataframe.columns:
            labels = session.dataframe["category_name"].values
            session.set_labels(labels)
        elif "category" in session.dataframe.columns:
            labels = session.dataframe["category"].values
            session.set_labels(labels)
            
        non_zero = int(X.nnz)
        total_elements = int(X.shape[0] * X.shape[1])
        sparsity_pct = float(100 * (1 - non_zero / total_elements)) if total_elements > 0 else 0.0
        
        result_id = session.next_result_id("dtm")
        summary = {
            "result_id": result_id,
            "n_documents": int(X.shape[0]),
            "n_terms": int(X.shape[1]),
            "vocabulary": feature_names,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct
        }
        
        session.store_result(
            tool_name="build_dtm_tool",
            args={"ngram_range": ngram_range, "max_features": max_features},
            summary=summary,
            full_report={"feature_matrix": X, "feature_names": feature_names}
        )
        
        return summary

    @tool
    def term_frequency_tool() -> dict:
        """Aggregates total term frequencies across all documents from session.feature_matrix and session.feature_names.
        
        Returns:
            dict: Summary containing term frequencies dictionary and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}
            
        X = session.feature_matrix
        feature_names = session.feature_names
        
        term_sums = np.array(X.sum(axis=0)).flatten()
        frequencies = {name: int(count) for name, count in zip(feature_names, term_sums)}
        
        result_id = session.next_result_id("term_frequency")
        summary = {
            "result_id": result_id,
            "frequencies": frequencies
        }
        
        session.store_result(
            tool_name="term_frequency_tool",
            args={},
            summary=summary,
            full_report={"frequencies": frequencies}
        )
        
        return summary

    @tool
    def dtm_heatmap_tool(n_terms: int = 20, n_documents: int = 20) -> dict:
        """Generates a heatmap of a raw positional slice of the global document-term matrix.
        
        Args:
            n_terms (int): Number of terms (columns) to include. Defaults to 20.
            n_documents (int): Number of documents (rows) to include. Defaults to 20.
            
        Returns:
            dict: Summary containing the sliced matrix as nested lists, terms, document labels, and result_id.
        """
        if session.feature_matrix is None or session.feature_names is None:
            return {"error": "No document-term matrix found. Please run build_dtm_tool first."}
            
        X = session.feature_matrix
        feature_names = session.feature_names
        
        n_docs_slice = min(n_documents, X.shape[0])
        n_terms_slice = min(n_terms, X.shape[1])
        
        sliced_X = X[0:n_docs_slice, 0:n_terms_slice].toarray()
        sliced_terms = feature_names[0:n_terms_slice]
        
        # Get document labels if available in session or dataframe
        document_labels = []
        if session.labels is not None:
            document_labels = [str(l) for l in session.labels[:n_docs_slice]]
        elif session.dataframe is not None:
            if "category_name" in session.dataframe.columns:
                document_labels = session.dataframe["category_name"].iloc[:n_docs_slice].astype(str).tolist()
            elif "category" in session.dataframe.columns:
                document_labels = session.dataframe["category"].iloc[:n_docs_slice].astype(str).tolist()
                
        # Build heatmap
        plt.figure(figsize=(max(6, n_terms_slice * 0.8), max(5, n_docs_slice * 0.5)))
        sns.heatmap(sliced_X, annot=True, fmt="d", cmap="Blues",
                    xticklabels=sliced_terms, yticklabels=document_labels if document_labels else list(range(n_docs_slice)))
        plt.title("Document-Term Matrix Heatmap")
        plt.xlabel("Terms")
        plt.ylabel("Documents")
        
        fig = plt.gcf()
        session.pending_figure = fig
        plt.close(fig)
        
        matrix_list = sliced_X.astype(int).tolist()
        
        result_id = session.next_result_id("dtm_heatmap")
        summary = {
            "result_id": result_id,
            "n_terms": n_terms_slice,
            "n_documents": n_docs_slice,
            "terms": sliced_terms,
            "document_labels": document_labels,
            "matrix": matrix_list
        }
        
        session.store_result(
            tool_name="dtm_heatmap_tool",
            args={"n_terms": n_terms, "n_documents": n_documents},
            summary=summary,
            full_report={"matrix": matrix_list, "terms": sliced_terms, "document_labels": document_labels}
        )
        
        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]

from langchain_core.tools import tool
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from PAMI.frequentPattern.basic import FPGrowth
from PAMI.frequentPattern.topk import FAE
from PAMI.frequentPattern.maximal import MaxFPGrowth
from PAMI.extras.convert.DF2DB import DF2DB
import tempfile
import os

def make_tools(session):
    @tool
    def mine_patterns_tool(
        category_name: str,
        filtering_method: str,
        algorithm: str = "fpgrowth",
        min_sup: int = 1,
        k: int = 5
    ) -> dict:
        """Mine frequent patterns from a category's documents using a specified vocabulary filter and mining algorithm.

        Args:
            category_name (str): The category name to filter and mine.
            filtering_method (str): Filtering approach to apply before mining ("variance", "tfidf", or "term_frequency").
            algorithm (str): Mining algorithm to run ("fpgrowth", "topk", or "maxfpgrowth", default "fpgrowth").
            min_sup (int): Minimum support threshold for fpgrowth or maxfpgrowth (default 1).
            k (int): Number of top-k patterns to keep for topk algorithm (default 5).

        Returns:
            dict: Summary of mined patterns or a note if no terms survived filtering.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}

        # 1. Filter session.dataframe to category_name and build fresh per-category CountVectorizer
        df_cat = session.dataframe[session.dataframe["category_name"] == category_name].copy()
        if df_cat.empty:
            return {"patterns": [], "note": f"No documents found for category '{category_name}'"}

        texts = df_cat["text"].fillna("").tolist()
        
        vectorizer = CountVectorizer(stop_words="english")
        try:
            X_cat = vectorizer.fit_transform(texts)
        except ValueError:
            return {"patterns": [], "note": "no terms survived filtering"}

        feature_names = np.array(vectorizer.get_feature_names_out())
        if len(feature_names) == 0:
            return {"patterns": [], "note": "no terms survived filtering"}

        # 2. Apply chosen filtering method
        # variance / term_frequency keep middle 90% (5th-95th percentile)
        # tfidf keeps top 80% (>= 20th percentile) by mean TF-IDF score
        if filtering_method in ["variance", "term_frequency"]:
            if filtering_method == "variance":
                # variance across rows
                scores = np.var(X_cat.toarray(), axis=0)
            else:
                # column sums (term frequency)
                scores = np.array(X_cat.sum(axis=0)).flatten()
            
            if len(scores) == 0:
                return {"patterns": [], "note": "no terms survived filtering"}

            p5 = np.percentile(scores, 5)
            p95 = np.percentile(scores, 95)
            
            mask = (scores >= p5) & (scores <= p95)
            kept_indices = np.where(mask)[0]
            
        elif filtering_method == "tfidf":
            transformer = TfidfTransformer()
            X_tfidf = transformer.fit_transform(X_cat)
            scores = np.array(X_tfidf.mean(axis=0)).flatten()
            
            if len(scores) == 0:
                return {"patterns": [], "note": "no terms survived filtering"}

            p20 = np.percentile(scores, 20)
            mask = scores >= p20
            kept_indices = np.where(mask)[0]
        else:
            return {"error": f"Unknown filtering method: {filtering_method}"}

        if len(kept_indices) == 0:
            return {"patterns": [], "note": "no terms survived filtering"}

        # Filter DTM columns and feature names
        X_filtered = X_cat[:, kept_indices]
        filtered_names = feature_names[kept_indices].tolist()

        dense_filtered = X_filtered.toarray()
        if dense_filtered.size == 0 or len(filtered_names) == 0:
            return {"patterns": [], "note": "no terms survived filtering"}

        # Build dense DataFrame with filtered_names as columns
        trans_df = pd.DataFrame(dense_filtered, columns=filtered_names)

        # Write out to a temporary transactional database file for PAMI via DF2DB
        db_fd, db_path = tempfile.mkstemp(suffix=".db")
        os.close(db_fd)

        try:
            obj = DF2DB(trans_df, DFtype="dense")
            obj.convert2TransactionalDatabase(db_path, ">=", 1)

            # 4. Run requested algorithm
            patterns_list = []
            alg_lower = algorithm.lower()
            if alg_lower == "fpgrowth":
                obj_mining = FPGrowth.FPGrowth(db_path, min_sup, sep="\t")
                obj_mining.mine()
                raw_patterns = obj_mining.getPatterns()
            elif alg_lower == "topk":
                obj_mining = FAE.FAE(db_path, k, sep="\t")
                obj_mining.mine()
                raw_patterns = obj_mining.getPatterns()
            elif alg_lower == "maxfpgrowth":
                obj_mining = MaxFPGrowth.MaxFPGrowth(db_path, min_sup, sep="\t")
                obj_mining.mine()
                raw_patterns = obj_mining.getPatterns()
            else:
                return {"error": f"Unknown algorithm: {algorithm}"}

            for pat, sup in raw_patterns.items():
                if isinstance(pat, str):
                    pat_list = [pat]
                elif isinstance(pat, (list, tuple)):
                    pat_list = list(pat)
                else:
                    pat_list = [str(pat)]
                patterns_list.append({"pattern": pat_list, "support": sup})
        finally:
            # Clean up temp db file
            if db_path and os.path.exists(db_path):
                os.remove(db_path)

        result_id = session.next_result_id("patterns")
        summary = {
            "result_id": result_id,
            "category_name": category_name,
            "filtering_method": filtering_method,
            "algorithm": algorithm,
            "n_patterns": len(patterns_list),
            "patterns": patterns_list
        }
        
        session.artifacts[f"patterns_{category_name}_{filtering_method}_{algorithm}"] = patterns_list
        session.store_result("mine_patterns_tool", {"category_name": category_name, "filtering_method": filtering_method, "algorithm": algorithm}, summary)

        return summary

    return [mine_patterns_tool]

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from langchain_core.tools import tool

def make_tools(session):
    @tool
    def tokenize_tool() -> dict:
        """Tokenizes each document's text in session.dataframe into unigrams using nltk.word_tokenize on lowercased text.
        
        Stores the resulting token lists in session.dataframe['unigrams'] and session.artifacts['tokens'],
        and records the result via session.store_result.
        
        Returns:
            dict: Summary containing the result_id and total documents tokenized.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
        
        import nltk
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt', quiet=True)
            
        try:
            nltk.data.find('tokenizers/punkt_tab')
        except LookupError:
            nltk.download('punkt_tab', quiet=True)

        tokens_list = []
        for text in session.dataframe["text"]:
            if pd.isna(text) or not isinstance(text, str) or text.strip() == "":
                tokens_list.append([])
            else:
                tokens_list.append(nltk.word_tokenize(text.lower()))
                
        session.dataframe["unigrams"] = tokens_list
        session.artifacts["tokens"] = tokens_list
        
        result_id = session.next_result_id("tokenize")
        summary = {
            "result_id": result_id,
            "n_documents": len(tokens_list)
        }
        
        session.store_result(
            tool_name="tokenize_tool",
            args={},
            summary=summary,
            full_report={"tokens": tokens_list}
        )
        
        return summary

    @tool
    def list_files_tool(subdirectory: str = "") -> dict:
        """Lists files and folders in the specified workspace or repository subdirectory, helping discover datasets before loading them.
        
        Args:
            subdirectory (str): Subdirectory path to list contents of, relative to the repository root. Defaults to "".
            
        Returns:
            dict: Summary containing directories, files, and result_id.
        """
        target_dir = subdirectory if subdirectory else "."
        
        if not os.path.exists(target_dir):
            return {"error": f"Directory '{subdirectory}' does not exist."}
            
        entries = os.listdir(target_dir)
        directories = sorted([e for e in entries if os.path.isdir(os.path.join(target_dir, e)) and not e.startswith('.')])
        files = sorted([e for e in entries if os.path.isfile(os.path.join(target_dir, e)) and not e.startswith('.')])
        
        result_id = session.next_result_id("list_files")
        summary = {
            "result_id": result_id,
            "subdirectory": subdirectory,
            "directories": directories,
            "files": files
        }
        
        session.store_result(
            tool_name="list_files_tool",
            args={"subdirectory": subdirectory},
            summary=summary
        )
        
        return summary

    @tool
    def inspect_data_tool(n_rows: int = 5) -> dict:
        """Inspects the working dataset by previewing a small number of rows from session.dataframe.
        
        Args:
            n_rows (int): Number of rows to inspect from the top of the dataframe. Must be greater than 0. Defaults to 5.
            
        Returns:
            dict: Summary containing preview rows as records, total row/column count, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
            
        if n_rows <= 0:
            return {"error": "Parameter n_rows must be greater than 0."}
            
        sample_df = session.dataframe.head(n_rows)
        records = sample_df.to_dict(orient="records")
        
        result_id = session.next_result_id("inspect")
        summary = {
            "result_id": result_id,
            "shape": list(session.dataframe.shape),
            "columns": list(session.dataframe.columns),
            "preview_rows": records
        }
        
        session.store_result(
            tool_name="inspect_data_tool",
            args={"n_rows": n_rows},
            summary=summary,
            full_report={"preview": sample_df}
        )
        
        return summary

    @tool
    def check_missing_tool() -> dict:
        """Checks session.dataframe for missing or empty text values, reporting the count of missing rows.
        
        Returns:
            dict: Summary containing the count of missing rows and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
            
        df = session.dataframe
        missing_mask = df["text"].apply(lambda t: pd.isna(t) or not isinstance(t, str) or t.strip() == "")
        missing_count = int(missing_mask.sum())
        missing_indices = df.index[missing_mask].tolist()
        
        result_id = session.next_result_id("check_missing")
        summary = {
            "result_id": result_id,
            "missing_count": missing_count,
            "missing_indices": missing_indices
        }
        
        session.store_result(
            tool_name="check_missing_tool",
            args={},
            summary=summary,
            full_report={"missing_mask": missing_mask}
        )
        
        return summary

    @tool
    def check_duplicates_tool(drop: bool = False) -> dict:
        """Checks for full duplicate rows in session.dataframe. Optionally drops duplicates using keep=False (removing all copies of duplicated rows).
        
        Args:
            drop (bool): Whether to drop duplicate rows from session.dataframe using keep=False. Defaults to False.
            
        Returns:
            dict: Summary containing duplicate count, duplicate indices, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
            
        df = session.dataframe
        dup_mask = df.duplicated(keep=False)
        duplicate_count = int(df.duplicated().sum())
        duplicate_indices = df.index[dup_mask].tolist()
        
        if drop:
            df.drop_duplicates(keep=False, inplace=True)
            df.reset_index(drop=True, inplace=True)
            
        result_id = session.next_result_id("check_duplicates")
        summary = {
            "result_id": result_id,
            "duplicate_count": duplicate_count,
            "duplicate_indices": duplicate_indices,
            "dropped": drop,
            "current_n_rows": len(df)
        }
        
        session.store_result(
            tool_name="check_duplicates_tool",
            args={"drop": drop},
            summary=summary,
            full_report={"dup_mask": dup_mask}
        )
        
        return summary

    @tool
    def sample_data_tool(n: int = 5, random_state: int = None) -> dict:
        """Subsamples session.dataframe using pandas DataFrame.sample.
        
        Args:
            n (int): Number of rows to sample. Defaults to 5.
            random_state (int, optional): Random seed for reproducibility. Defaults to None.
            
        Returns:
            dict: Summary containing sampled row indices and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
            
        df = session.dataframe
        if n > len(df):
            return {"error": f"Requested sample size {n} exceeds number of rows {len(df)}."}
            
        sampled_df = df.sample(n=n, random_state=random_state)
        sampled_indices = sampled_df.index.tolist()
        
        result_id = session.next_result_id("sample")
        summary = {
            "result_id": result_id,
            "n": n,
            "random_state": random_state,
            "sampled_indices": sampled_indices,
            "preview_rows": sampled_df.to_dict(orient="records")
        }
        
        session.store_result(
            tool_name="sample_data_tool",
            args={"n": n, "random_state": random_state},
            summary=summary,
            full_report={"sampled_dataframe": sampled_df}
        )
        
        return summary

    @tool
    def describe_data_tool() -> dict:
        """Computes descriptive statistics on document text length (character count) overall and per category,
        and generates a box plot set on session.pending_figure.
        
        Returns:
            dict: Summary containing overall describe stats, per-category stats, and result_id.
        """
        if session.dataframe is None:
            return {"error": "No dataset loaded. Please load a dataset first."}
            
        df = session.dataframe.copy()
        df["text_length"] = df["text"].apply(len)
        
        overall_desc = df["text_length"].describe().to_dict()
        
        per_category_desc = {}
        group_col = "category_name" if "category_name" in df.columns else "category"
        for cat, group in df.groupby(group_col):
            per_category_desc[str(cat)] = group["text_length"].describe().to_dict()
                
        # Build box plot of text_length grouped by category_name (matching Master's call)
        plt.figure(figsize=(8, 5))
        sns.boxplot(x="category_name" if "category_name" in df.columns else "category", y="text_length", data=df)
        plt.title("Document Text Length by Category")
        plt.xlabel("Category")
        plt.ylabel("Text Length (Characters)")
        
        fig = plt.gcf()
        session.pending_figure = fig
        plt.close(fig)
        
        result_id = session.next_result_id("describe")
        summary = {
            "result_id": result_id,
            "overall": overall_desc,
            "per_category": per_category_desc
        }
        
        session.store_result(
            tool_name="describe_data_tool",
            args={},
            summary=summary,
            full_report={"dataframe": df}
        )
        
        return summary

    return [tokenize_tool, list_files_tool, inspect_data_tool, check_missing_tool, check_duplicates_tool, sample_data_tool, describe_data_tool]

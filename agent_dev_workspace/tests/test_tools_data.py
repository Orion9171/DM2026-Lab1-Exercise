from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools
import pytest

def test_tokenize_tool():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    # Initialize and invoke tokenize_tool from tools_data.py
    data_tools = make_tools(session)
    tokenize_tool = next(t for t in data_tools if t.name == "tokenize_tool")
    result = tokenize_tool.invoke({})
    
    assert result["n_documents"] == 8
    
    tokens = session.artifacts["tokens"]
    
    # Expected tokens from TEST_FIXTURE.md #10
    expected_tokens = [
        ["always", "alpha", "alpha", "alpha", "beta"],
        ["always", "alpha", "alpha", "beta"],
        ["always", "alpha", "beta"],
        ["always", "gamma", "gamma", "gamma", "delta"],
        ["always", "gamma", "gamma", "delta"],
        ["always", "gamma", "delta"],
        ["always", "gamma", "delta"],
        []
    ]
    
    assert len(tokens) == len(expected_tokens)
    for i, (actual, expected) in enumerate(zip(tokens, expected_tokens)):
        assert actual == expected, f"Mismatch at document {i}: got {actual}, expected {expected}"

def test_list_files_tool():
    session = SessionState()
    data_tools = make_tools(session)
    list_files = next(t for t in data_tools if t.name == "list_files_tool")
    
    result = list_files.invoke({"subdirectory": "newdataset"})
    
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]

def test_inspect_data_tool():
    session = SessionState()
    
    # Load dataset using premade load_dataset_tool
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    inspect_data = next(t for t in data_tools if t.name == "inspect_data_tool")
    
    # Test normal inspection with n_rows=3
    result = inspect_data.invoke({"n_rows": 3})
    assert "error" not in result
    assert len(result["preview_rows"]) == 3
    assert result["shape"][0] == 8  # 8 rows
    assert "text" in result["columns"]
    assert "category" in result["columns"]
    assert "category_name" in result["columns"]
    
    # Test invalid n_rows=0 validation
    error_result = inspect_data.invoke({"n_rows": 0})
    assert "error" in error_result

def test_check_missing_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    check_missing = next(t for t in data_tools if t.name == "check_missing_tool")
    
    result = check_missing.invoke({})
    
    # Known Answer #2: Exactly 1 row has missing/empty text (the last row)
    assert result["missing_count"] == 1
    assert result["missing_indices"] == [7]

def test_check_duplicates_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    check_duplicates = next(t for t in data_tools if t.name == "check_duplicates_tool")
    
    # Test checking duplicates without dropping (Known Answer #3: exactly 1 duplicate, row 6 is copy of row 5)
    result = check_duplicates.invoke({"drop": False})
    assert result["duplicate_count"] == 1
    assert result["duplicate_indices"] == [5, 6]
    
    # Test dropping duplicates with drop=True (Known Answer #3: dropping with keep=False leaves exactly 6 rows: rows 5 and 6 removed)
    result_drop = check_duplicates.invoke({"drop": True})
    assert result_drop["current_n_rows"] == 6
    assert len(session.dataframe) == 6

def test_sample_data_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    sample_data = next(t for t in data_tools if t.name == "sample_data_tool")
    
    # Known Answer #13: Call with n=4, random_state=42 returns exactly rows [1, 5, 0, 7]
    result = sample_data.invoke({"n": 4, "random_state": 42})
    assert result["sampled_indices"] == [1, 5, 0, 7]

def test_describe_data_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    data_tools = make_tools(session)
    describe_data = next(t for t in data_tools if t.name == "describe_data_tool")
    
    result = describe_data.invoke({})
    
    # Known Answer #18: Overall .describe() stats
    overall = result["overall"]
    assert overall["count"] == 8
    assert overall["mean"] == pytest.approx(19.875, abs=1e-3)
    assert overall["std"] == pytest.approx(9.4330, abs=1e-3)
    assert overall["min"] == 0
    assert overall["25%"] == pytest.approx(17.75, abs=1e-3)
    assert overall["50%"] == pytest.approx(20.5, abs=1e-3)
    assert overall["75%"] == pytest.approx(25.25, abs=1e-3)
    assert overall["max"] == 30
    
    # Per category catA
    catA = result["per_category"]["catA"]
    assert catA["count"] == 4
    assert catA["mean"] == pytest.approx(17.25, abs=1e-3)
    assert catA["std"] == pytest.approx(12.5, abs=1e-3)
    assert catA["min"] == 0
    assert catA["25%"] == pytest.approx(12.75, abs=1e-3)
    assert catA["50%"] == pytest.approx(20.0, abs=1e-3)
    assert catA["75%"] == pytest.approx(24.5, abs=1e-3)
    assert catA["max"] == 29
    
    # Per category catB
    catB = result["per_category"]["catB"]
    assert catB["count"] == 4
    assert catB["mean"] == pytest.approx(22.5, abs=1e-3)
    assert catB["std"] == pytest.approx(5.7446, abs=1e-3)
    assert catB["min"] == 18
    assert catB["25%"] == pytest.approx(18.0, abs=1e-3)
    assert catB["50%"] == pytest.approx(21.0, abs=1e-3)
    assert catB["75%"] == pytest.approx(25.5, abs=1e-3)
    assert catB["max"] == 30
    
    # Verify plotting requirement: session.pending_figure is set
    assert session.pending_figure is not None

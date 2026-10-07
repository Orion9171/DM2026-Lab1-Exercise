from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools
import pytest

def test_build_dtm_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    
    result = build_dtm.invoke({})
    
    # Known Answer #4: Vocabulary ['alpha', 'always', 'beta', 'delta', 'gamma'] (5 terms)
    # Sparsity: 47.5%
    assert result["vocabulary"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result["n_terms"] == 5
    assert result["non_zero"] == 21
    assert result["sparsity_pct"] == pytest.approx(47.5, abs=1e-3)
    
    assert session.feature_names == ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts

def test_term_frequency_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    term_freq = next(t for t in dtm_tools if t.name == "term_frequency_tool")
    result = term_freq.invoke({})
    
    # Known Answer #14: Total frequencies
    # alpha: 6, always: 7, beta: 3, delta: 4, gamma: 7
    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7
    }
    
    assert result["frequencies"] == expected_frequencies

def test_dtm_heatmap_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")
    
    # Test full matrix slice (defaults n_terms=20, n_documents=20)
    result_full = heatmap_tool.invoke({})
    expected_full_matrix = [
        [3, 1, 1, 0, 0],
        [2, 1, 1, 0, 0],
        [1, 1, 1, 0, 0],
        [0, 1, 0, 1, 3],
        [0, 1, 0, 1, 2],
        [0, 1, 0, 1, 1],
        [0, 1, 0, 1, 1],
        [0, 0, 0, 0, 0]
    ]
    assert result_full["matrix"] == expected_full_matrix
    assert result_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]
    assert result_full["n_documents"] == 8
    assert session.pending_figure is not None
    
    # Test specific corner slice n_terms=2, n_documents=3
    result_slice = heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    expected_corner = [
        [3, 1],
        [2, 1],
        [1, 1]
    ]
    assert result_slice["matrix"] == expected_corner
    assert result_slice["terms"] == ["alpha", "always"]
    assert result_slice["n_documents"] == 3

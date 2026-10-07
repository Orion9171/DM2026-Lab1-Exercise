import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_exploration import make_tools as make_exploration_tools

def setup_session_with_dtm():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    return session

def test_cosine_similarity_known_answers():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    cosine_sim = next(t for t in exploration_tools if t.name == "cosine_similarity_tool")

    # Known Answer #16: 0 vs 1 (both catA, near-identical text) -> ~0.9847
    res_0_1 = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 1})
    assert "error" not in res_0_1
    assert res_0_1.get("cosine_similarity") == pytest.approx(0.9847, abs=1e-3)

    # Known Answer #16: 0 vs 3 (catA vs catB, share only "always") -> ~0.0909
    res_0_3 = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 3})
    assert "error" not in res_0_3
    assert res_0_3.get("cosine_similarity") == pytest.approx(0.0909, abs=1e-3)

    # Known Answer #16: 3 vs 4 (both catB, near-identical text) -> ~0.9847
    res_3_4 = cosine_sim.invoke({"doc_index_1": 3, "doc_index_2": 4})
    assert "error" not in res_3_4
    assert res_3_4.get("cosine_similarity") == pytest.approx(0.9847, abs=1e-3)

    # Known Answer #16: 0 vs 7 (catA vs empty-text row) -> 0.0
    res_0_7 = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 7})
    assert "error" not in res_0_7
    assert res_0_7.get("cosine_similarity") == pytest.approx(0.0, abs=1e-3)

    # Out of bounds check
    res_err = cosine_sim.invoke({"doc_index_1": 0, "doc_index_2": 100})
    assert "error" in res_err

def test_feature_correlation_matrix_known_answers():
    session = setup_session_with_dtm()
    exploration_tools = make_exploration_tools(session)
    feat_corr = next(t for t in exploration_tools if t.name == "feature_correlation_matrix_tool")

    session.pending_figure = None
    res = feat_corr.invoke({})
    assert "error" not in res

    # Known Answer #19: Feature order sorted by variance descending: alpha, gamma, delta, beta, always
    feature_names = res.get("feature_names")
    expected_features = ["alpha", "gamma", "delta", "beta", "always"]
    assert feature_names == expected_features

    # Known Answer #19: Full correlation matrix
    corr_matrix = np.array(res.get("correlation_matrix"))
    expected_corr = np.array([
        [1.0000, -0.5718, -0.6882,  0.8885,  0.2601],
        [-0.5718, 1.0000,  0.8307, -0.6435,  0.3140],
        [-0.6882, 0.8307,  1.0000, -0.7746,  0.3780],
        [0.8885, -0.6435, -0.7746,  1.0000,  0.2928],
        [0.2601,  0.3140,  0.3780,  0.2928,  1.0000]
    ])
    assert np.allclose(corr_matrix, expected_corr, atol=1e-3)
    assert session.pending_figure is not None

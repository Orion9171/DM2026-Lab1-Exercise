from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools
import pytest

def test_variance_filter_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    filtering_tools = make_filtering_tools(session)
    variance_filter = next(t for t in filtering_tools if t.name == "variance_filter_tool")
    
    result = variance_filter.invoke({"threshold": 0.15})
    
    assert "always" in result["removed_terms"]
    assert sorted(result["kept_terms"]) == sorted(["alpha", "beta", "delta", "gamma"])
    
    variances = result["term_variances"]
    assert variances["alpha"] == pytest.approx(1.1875, abs=1e-3)
    assert variances["always"] == pytest.approx(0.109375, abs=1e-3)
    assert variances["beta"] == pytest.approx(0.234375, abs=1e-3)
    assert variances["delta"] == pytest.approx(0.25, abs=1e-3)
    assert variances["gamma"] == pytest.approx(1.109375, abs=1e-3)

def test_pearson_filter_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    filtering_tools = make_filtering_tools(session)
    pearson_filter = next(t for t in filtering_tools if t.name == "pearson_filter_tool")
    
    result = pearson_filter.invoke({"target_class": "catB"})
    corrs = result["correlations"]
    
    assert corrs["alpha"] == pytest.approx(-0.6882, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.8307, abs=1e-3)

def test_spearman_filter_tool():
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    
    dtm_tools = make_dtm_tools(session)
    build_dtm = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm.invoke({})
    
    filtering_tools = make_filtering_tools(session)
    spearman_filter = next(t for t in filtering_tools if t.name == "spearman_filter_tool")
    
    # Test target_class="catB" (Known Answer #6)
    result = spearman_filter.invoke({"target_class": "catB"})
    corrs = result["correlations"]
    
    # Known Answer #6 Spearman r:
    # alpha: -0.7500, always: 0.3780, beta: -0.7746, delta: 1.0000, gamma: 0.9363
    assert corrs["alpha"] == pytest.approx(-0.7500, abs=1e-3)
    assert corrs["always"] == pytest.approx(0.3780, abs=1e-3)
    assert corrs["beta"] == pytest.approx(-0.7746, abs=1e-3)
    assert corrs["delta"] == pytest.approx(1.0000, abs=1e-3)
    assert corrs["gamma"] == pytest.approx(0.9363, abs=1e-3)

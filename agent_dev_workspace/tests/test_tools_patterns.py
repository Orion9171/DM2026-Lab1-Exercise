import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_pattern_tools

def setup_session():
    session = SessionState()
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })
    return session

def test_mine_patterns_known_answers():
    session = setup_session()
    pattern_tools = make_pattern_tools(session)
    mine_patterns = next(t for t in pattern_tools if t.name == "mine_patterns_tool")

    # Known answer #17: variance and term_frequency filtering methods keep zero terms for both categories
    res_var = mine_patterns.invoke({"category_name": "catA", "filtering_method": "variance"})
    assert res_var.get("note") == "no terms survived filtering"
    assert res_var.get("patterns") == []

    res_tf = mine_patterns.invoke({"category_name": "catB", "filtering_method": "term_frequency"})
    assert res_tf.get("note") == "no terms survived filtering"
    assert res_tf.get("patterns") == []

    # Known answer #17: tfidf filtering + fpgrowth for catA (expect pattern ['alpha'] with support 3)
    res_catA_fp = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 3})
    patterns_a = res_catA_fp.get("patterns", [])
    assert len(patterns_a) == 1
    assert set(patterns_a[0]["pattern"]) == {"alpha"}
    assert patterns_a[0]["support"] == 3

    # Known answer #17: tfidf filtering + fpgrowth for catB (expect pattern ['gamma'] with support 4)
    res_catB_fp = mine_patterns.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "fpgrowth", "min_sup": 4})
    patterns_b = res_catB_fp.get("patterns", [])
    assert len(patterns_b) == 1
    assert set(patterns_b[0]["pattern"]) == {"gamma"}
    assert patterns_b[0]["support"] == 4

    # Known answer #20: topk algorithm (at k=1, k=2, k=3 alike, both categories return exactly one pattern)
    res_topk_a = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "topk", "k": 1})
    assert len(res_topk_a.get("patterns", [])) == 1
    assert set(res_topk_a.get("patterns", [])[0]["pattern"]) == {"alpha"}

    res_topk_b = mine_patterns.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "topk", "k": 2})
    assert len(res_topk_b.get("patterns", [])) == 1
    assert set(res_topk_b.get("patterns", [])[0]["pattern"]) == {"gamma"}

    # Known answer #20: maxfpgrowth returns zero patterns for both categories at min_sup=1 and min_sup=4
    res_maxfp_a = mine_patterns.invoke({"category_name": "catA", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": 1})
    assert res_maxfp_a.get("patterns", []) == []

    res_maxfp_b = mine_patterns.invoke({"category_name": "catB", "filtering_method": "tfidf", "algorithm": "maxfpgrowth", "min_sup": 4})
    assert res_maxfp_b.get("patterns", []) == []

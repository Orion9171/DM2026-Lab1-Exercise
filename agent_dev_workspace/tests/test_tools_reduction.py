import pytest
import numpy as np
from scipy.spatial.distance import pdist
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools

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

def test_reduce_dimensions_known_answers():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    reduce_dims = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    # Known answer #7: PCA (exact deterministic on this input)
    session.pending_figure = None
    res_pca = reduce_dims.invoke({"method": "pca"})
    assert res_pca.get("method") == "pca"
    coords_pca = np.array(res_pca.get("coordinates"))
    
    expected_pca = np.array([
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822]
    ])
    assert np.allclose(coords_pca, expected_pca, atol=1e-3)

    ev_ratio = res_pca.get("explained_variance_ratio")
    assert np.allclose(ev_ratio, [0.7532, 0.1965], atol=1e-3)
    assert session.pending_figure is not None

    # Known answer #8: t-SNE with perplexity=2, random_state=42.
    session.pending_figure = None
    res_tsne1 = reduce_dims.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    assert res_tsne1.get("method") == "tsne"
    coords_tsne1 = np.array(res_tsne1.get("coordinates"))
    assert coords_tsne1.shape == (8, 2)
    assert np.isfinite(coords_tsne1).all()
    assert session.pending_figure is not None

    # Check determinism with same random_state
    session.pending_figure = None
    res_tsne2 = reduce_dims.invoke({"method": "tsne", "perplexity": 2.0, "random_state": 42})
    coords_tsne2 = np.array(res_tsne2.get("coordinates"))
    assert np.allclose(coords_tsne1, coords_tsne2)

    # Known answer #9: UMAP with n_neighbors=3, random_state=42.
    session.pending_figure = None
    res_umap1 = reduce_dims.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    assert res_umap1.get("method") == "umap"
    coords_umap1 = np.array(res_umap1.get("coordinates"))
    assert coords_umap1.shape == (8, 2)
    assert np.isfinite(coords_umap1).all()
    assert session.pending_figure is not None

    # Check determinism with same random_state
    session.pending_figure = None
    res_umap2 = reduce_dims.invoke({"method": "umap", "n_neighbors": 3, "random_state": 42})
    coords_umap2 = np.array(res_umap2.get("coordinates"))
    assert np.allclose(coords_umap1, coords_umap2)

    # Pairwise distance correlation against Known Answer #9
    expected_umap = np.array([
        [6.6811, -6.8218],
        [6.9907, -7.2245],
        [7.7022, -7.3712],
        [10.2228, -6.2310],
        [9.8759, -5.6537],
        [9.3831, -6.5748],
        [9.2153, -5.8297],
        [8.5860, -7.2622]
    ])
    dist_actual = pdist(coords_umap1)
    dist_expected = pdist(expected_umap)
    corr = np.corrcoef(dist_actual, dist_expected)[0, 1]
    assert corr >= 0.98

def test_binarize_labels_tool():
    session = setup_session_with_dtm()
    reduction_tools = make_reduction_tools(session)
    binarize_labels = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels.invoke({})
    assert "error" not in res

    # Verify classes count is 2 (catA and catB)
    classes = res.get("classes")
    assert len(classes) == 2

    # Verify output shape is (8, 2)
    shape = res.get("shape")
    assert shape == [8, 2]

    matrix = np.array(res.get("binary_matrix"))
    assert matrix.shape == (8, 2)

    # Every row contains exactly one 1 (one-hot encoding)
    row_sums = matrix.sum(axis=1)
    assert np.all(row_sums == 1)

    # Check that identical labels receive identical encodings
    # Rows 5 and 6 are both catB (same label category)
    assert np.array_equal(matrix[5], matrix[6])
    # Rows 0 and 3 are catA and catB respectively, should differ
    assert not np.array_equal(matrix[0], matrix[3])

    # Verify stored artifact matches returned binary matrix
    stored_artifact = session.artifacts.get("binarized_labels")
    assert stored_artifact is not None
    assert np.array_equal(stored_artifact, matrix)

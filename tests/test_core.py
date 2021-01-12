import pytest
import pandas as pd

import tahini.core


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # can only pass empty or index and/or data or size
    (
        [],
        dict(index=[], data=[], size=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'index', 'data', 'index' and 'data' or 'size'",
    ),
    # can only pass empty or index and/or data or size
    (
        [],
        dict(index=[], size=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'index', 'data', 'index' and 'data' or 'size'",
    ),
    # can only pass empty or index and/or data or size
    (
        [],
        dict(data=[], size=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'index', 'data', 'index' and 'data' or 'size'",
    ),
    # size errors are driven by range(stop=size)
    ([], dict(size=0.5), TypeError, "'float' object cannot be interpreted as an integer"),
    # if data is a data_frame then index has to be None because no simple way to set index to index without taking into
    # account other cases
    (
        [],
        dict(index=[], data=pd.DataFrame()),
        ValueError,
        "If input 'data' is 'pandas.DataFrame' then input 'index' has to be 'None' for initializing 'Nodes'",
    ),
])
def test_nodes_init_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.core.Nodes(*args, **kwargs)
    assert e.value.args[0] == message_error


def get_data_nodes(name_index='node', *args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(*args, **kwargs).rename_axis(index=name_index)


@pytest.mark.parametrize('args, kwargs, expected', [
    # inputs empty
    ([], dict(), get_data_nodes()),
    # args - nodes empty
    ([[]], dict(), get_data_nodes()),
    # args - data empty
    ([None, []], dict(), get_data_nodes()),
    # args - size
    ([None, None, 1], dict(), get_data_nodes(index=range(1))),
    # nodes empty
    ([], dict(index=[]), get_data_nodes()),
    # data empty
    ([], dict(data=[]), get_data_nodes()),
    # empty nodes and data
    ([], dict(index=[], data=[]), get_data_nodes()),
    # size zero
    ([], dict(size=0), get_data_nodes(index=range(0))),
    # size
    ([], dict(size=1), get_data_nodes(index=range(1))),
    # size negative
    ([], dict(size=-1), get_data_nodes(index=range(-1))),
    # nodes input it's own class
    ([], dict(index=tahini.core.Nodes()), get_data_nodes()),
])
def test_nodes_init(args, kwargs, expected):
    nodes = tahini.core.Nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes.data, expected)


def get_data_edges(*args, index=None, **kwargs):
    if index is None:
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['node_1', 'node_2'])
    else:
        index = pd.MultiIndex.from_tuples(index, names=['node_1', 'node_2'])
    return pd.DataFrame(*args, index=index, **kwargs)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(), get_data_edges()),
    # index list of tuples
    ([], dict(index=[(0, 1), (1, 2)]), get_data_edges(index=[(0, 1), (1, 2)])),
    # index list of lists
    ([], dict(index=[[0, 1], [1, 2]]), get_data_edges(index=[(0, 1), (1, 2)])),
    # index and data dict
    (
        [],
        dict(index=[[0, 1], [1, 2]], data=dict(value=['a', 'b'])),
        get_data_edges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
    # data_frame
    (
        [],
        dict(data=pd.DataFrame(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)])),
        get_data_edges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
    # idemponent index empty
    ([], dict(index=tahini.core.Edges()), get_data_edges()),
    # idemponent index non empty
    ([], dict(index=tahini.core.Edges(index=[(0, 1), (1, 2)])), get_data_edges(index=[(0, 1), (1, 2)])),
    # idemponent index and data non empty
    (
        [],
        dict(index=tahini.core.Edges(index=[(0, 1), (1, 2)], data=dict(value=['a', 'b']))),
        get_data_edges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
])
def test_edges_init(args, kwargs, expected):
    edges = tahini.core.Edges(*args, **kwargs)
    pd.testing.assert_frame_equal(edges.data, expected)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.Edges(), [], dict(), get_data_nodes()),
    # non empty
    (tahini.core.Edges(index=[(0, 1), (1, 2)]), [], dict(), get_data_nodes(index=[0, 1, 2])),
])
def test_edges_get_nodes(edges, args, kwargs, expected):
    nodes = edges.get_nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes.data, expected)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.Edges(), [], dict(), get_data_edges()),
    # non empty edges
    (tahini.core.Edges(index=[(0, 1), (1, 2)]), [], dict(), get_data_edges(index=[(0, 1), (1, 2)])),
    # non empty nodes
    (tahini.core.Edges(), [], dict(nodes=[0, 1]), get_data_edges()),
    # non empty both
    (tahini.core.Edges(index=[(0, 1), (1, 2)]), [], dict(nodes=[0, 1]), get_data_edges(index=[(0, 1)])),
])
def test_edges_get_nodes(edges, args, kwargs, expected):
    edges = edges.keep_nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(edges.data, expected)


@pytest.mark.parametrize('args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty
    ([], dict(), get_data_nodes(), get_data_edges()),
    # nodes
    ([], dict(nodes=[0, 1]), get_data_nodes(index=[0, 1]), get_data_edges()),
    # edges
    ([], dict(edges=[(0, 1), (1, 2)]), get_data_nodes(index=[0, 1, 2]), get_data_edges(index=[(0, 1), (1, 2)])),
    # nodes and edges
    (
        [],
        dict(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[(0, 1), (1, 2)]),
    ),
    # partial nodes and edges
    (
        [],
        dict(nodes=[0, 1], edges=[(0, 1), (1, 2)]),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[(0, 1), (1, 2)]),
    ),
    # degree
    ([], dict(degree=2), get_data_nodes(index=range(2)), get_data_edges()),
    # nodes data
    (
        [],
        dict(nodes_data=pd.DataFrame(data=dict(value=['a', 'b']))),
        get_data_nodes(data=dict(value=['a', 'b'])),
        get_data_edges(),
    ),
    # edges data
    (
        [],
        dict(edges_data=pd.DataFrame(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)])),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
])
def test_graph_init(args, kwargs, data_nodes_expected, data_edges_expected):
    graph = tahini.core.Graph(*args, **kwargs)
    pd.testing.assert_frame_equal(graph.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, edges, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty
    (
        tahini.core.Graph(),
        tahini.core.Edges(index=[[0, 1]]),
        [],
        dict(),
        get_data_nodes(index=[0, 1]),
        get_data_edges(index=[[0, 1]]),
    ),
    # non empty
    (
        tahini.core.Graph(nodes=[0]),
        tahini.core.Edges(index=[[1, 2]]),
        [],
        dict(),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[[1, 2]]),
    ),
])
def test_graph__update_nodes_from_edges(graph, edges, args, kwargs, data_nodes_expected, data_edges_expected):
    graph._edges = edges
    nodes = graph._update_nodes_from_edges(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, nodes, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty
    (
        tahini.core.Graph(),
        tahini.core.Nodes(index=[0, 1]),
        [],
        dict(),
        get_data_nodes(index=[0, 1]),
        get_data_edges(),
    ),
    # non empty
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        tahini.core.Nodes(index=[0, 1]),
        [],
        dict(),
        get_data_nodes(index=[0, 1]),
        get_data_edges(index=[[0, 1]]),
    ),
])
def test_graph__update_edges_from_nodes(graph, nodes, args, kwargs, data_nodes_expected, data_edges_expected):
    graph._nodes = nodes
    edges = graph._update_edges_from_nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(graph.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty
    (tahini.core.Graph(), [], dict(), get_data_nodes(), get_data_edges()),
    # non empty graph
    (tahini.core.Graph(nodes=[0, 1]), [], dict(), get_data_nodes(index=[0, 1]), get_data_edges()),
    # non empty nodes
    (tahini.core.Graph(), [], dict(nodes=[0, 1]), get_data_nodes(index=[0, 1]), get_data_edges()),
    # non empty both same
    (tahini.core.Graph(nodes=[0, 1]), [], dict(nodes=[0, 1]), get_data_nodes(index=[0, 1]), get_data_edges()),
    # non empty both different
    (tahini.core.Graph(nodes=[0, 1]), [], dict(nodes=[2]), get_data_nodes(index=[0, 1, 2]), get_data_edges()),
    # data
    (
        tahini.core.Graph(nodes=[0, 1]),
        [],
        dict(data=dict(value=['a', 'b'])),
        get_data_nodes(data=dict(value=['a', 'b']), index=[0, 1]),
        get_data_edges(),
    ),
])
def test_graph_update_nodes(graph, args, kwargs, data_nodes_expected, data_edges_expected):
    graph_updated = graph.update_nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(graph_updated.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph_updated.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty
    (tahini.core.Graph(), [], dict(), get_data_nodes(), get_data_edges()),
    # non empty graph
    (tahini.core.Graph(edges=[[0, 1]]), [], dict(), get_data_nodes(index=[0, 1]), get_data_edges(index=[[0, 1]])),
    # non empty edges
    (tahini.core.Graph(), [], dict(edges=[[0, 1]]), get_data_nodes(index=[0, 1]), get_data_edges(index=[[0, 1]])),
    # non empty both different
    (
        tahini.core.Graph(edges=[[0, 1]]),
        [],
        dict(edges=[[0, 1]]),
        get_data_nodes(index=[0, 1]),
        get_data_edges(index=[[0, 1]]),
    ),
    # non empty both different
    (
        tahini.core.Graph(edges=[[0, 1]]),
        [],
        dict(edges=[[1, 2]]),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[[0, 1], [1, 2]]),
    ),
    # data
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(edges=[[0, 1], [1, 2]], data=dict(value=['a', 'b'])),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(data=dict(value=['a', 'b']), index=[[0, 1], [1, 2]]),
    ),
])
def test_graph_update_edges(graph, args, kwargs, data_nodes_expected, data_edges_expected):
    graph_updated = graph.update_edges(*args, **kwargs)
    pd.testing.assert_frame_equal(graph_updated.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph_updated.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty
    (tahini.core.Graph(), [], dict(), get_data_nodes(), get_data_edges()),
])
def test_graph_update(graph, args, kwargs, data_nodes_expected, data_edges_expected):
    graph_updated = graph.update(*args, **kwargs)
    pd.testing.assert_frame_equal(graph_updated.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph_updated.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty all
    (tahini.core.Graph(), [], dict(), get_data_nodes(), get_data_edges()),
    # empty list
    (tahini.core.Graph(), [], dict(nodes=[]), get_data_nodes(), get_data_edges()),
    # empty inputs
    (tahini.core.Graph(nodes=[0, 1]), [], dict(), get_data_nodes(index=[0, 1]), get_data_edges()),
    # non empty
    (tahini.core.Graph(nodes=[0, 1]), [], dict(nodes=[0]), get_data_nodes(index=[1]), get_data_edges()),
    # non empty with removing edges
    (
        tahini.core.Graph(nodes=[0, 1], edges=[[0, 1]]),
        [],
        dict(nodes=[0]),
        get_data_nodes(index=[1]),
        get_data_edges(index=[[0, 1]]).drop(index=[(0, 1)]),
    ),
])
def test_graph_drop_nodes(graph, args, kwargs, data_nodes_expected, data_edges_expected):
    graph_dropped = graph.drop_nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(graph_dropped.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph_dropped.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty all
    (tahini.core.Graph(), [], dict(), get_data_nodes(), get_data_edges()),
    # empty list
    (tahini.core.Graph(), [], dict(edges=[]), get_data_nodes(), get_data_edges()),
    # empty inputs
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[[0, 1], [1, 2]]),
    ),
    # non empty
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(edges=[(0, 1)]),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[[1, 2]]),
    ),
])
def test_graph_drop_edges(graph, args, kwargs, data_nodes_expected, data_edges_expected):
    graph_dropped = graph.drop_edges(*args, **kwargs)
    pd.testing.assert_frame_equal(graph_dropped.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph_dropped.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, args, kwargs, data_nodes_expected, data_edges_expected', [
    # empty all
    (tahini.core.Graph(), [], dict(), get_data_nodes(), get_data_edges()),
    # empty lists
    (tahini.core.Graph(), [], dict(nodes=[], edges=[]), get_data_nodes(), get_data_edges()),
    # empty inputs
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(),
        get_data_nodes(index=[0, 1, 2]),
        get_data_edges(index=[[0, 1], [1, 2]]),
    ),
    # non empty
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2], [1, 3]]),
        [],
        dict(nodes=[0], edges=[(1, 2)]),
        get_data_nodes(index=[1, 2, 3]),
        get_data_edges(index=[[1, 3]]),
    ),
])
def test_graph_drop(graph, args, kwargs, data_nodes_expected, data_edges_expected):
    graph_dropped = graph.drop(*args, **kwargs)
    pd.testing.assert_frame_equal(graph_dropped.nodes.data, data_nodes_expected)
    pd.testing.assert_frame_equal(graph_dropped.edges.data, data_edges_expected)


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.Graph(), 0),
    # non empty
    (tahini.core.Graph(nodes=[0, 1]), 2),
    # degree
    (tahini.core.Graph(degree=3), 3)
])
def test_graph_degree(graph, expected):
    degree = graph.degree
    assert degree == expected

import pytest
import pandas as pd

import tahini.core
import tahini.testing


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # can only pass empty or index and/or data or order
    (
        [],
        dict(index=[], data=[], order=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'index', 'data', 'index' and 'data' or 'order'",
    ),
    # can only pass empty or index and/or data or order
    (
        [],
        dict(index=[], order=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'index', 'data', 'index' and 'data' or 'order'",
    ),
    # can only pass empty or index and/or data or order
    (
        [],
        dict(data=[], order=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'index', 'data', 'index' and 'data' or 'order'",
    ),
    # order errors are driven by range(stop=order)
    ([], dict(order=0.5), TypeError, "'float' object cannot be interpreted as an integer"),
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
    # args - order
    ([None, None, 1], dict(), get_data_nodes(index=range(1))),
    # nodes empty
    ([], dict(index=[]), get_data_nodes()),
    # data empty
    ([], dict(data=[]), get_data_nodes()),
    # empty nodes and data
    ([], dict(index=[], data=[]), get_data_nodes()),
    # order zero
    ([], dict(order=0), get_data_nodes(index=range(0))),
    # order
    ([], dict(order=1), get_data_nodes(index=range(1))),
    # order negative
    ([], dict(order=-1), get_data_nodes(index=range(-1))),
    # nodes input it's own class
    ([], dict(index=tahini.core.Nodes()), get_data_nodes()),
])
def test_nodes_init(args, kwargs, expected):
    nodes = tahini.core.Nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes.data, expected)


def get_data_edges(*args, index=None, **kwargs):
    if index is None:
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['node_0', 'node_1'])
    else:
        index = pd.MultiIndex.from_tuples(index, names=['node_0', 'node_1'])
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
    (tahini.core.Edges(), [], dict(), tahini.core.Nodes()),
    # non empty
    (tahini.core.Edges(index=[(0, 1), (1, 2)]), [], dict(), tahini.core.Nodes(index=[0, 1, 2])),
])
def test_edges_get_nodes(edges, args, kwargs, expected):
    nodes = edges.get_nodes(*args, **kwargs)
    tahini.testing.assert_nodes_equal(nodes, expected)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.Edges(), [], dict(), tahini.core.Edges()),
    # non empty edges
    (tahini.core.Edges(index=[(0, 1), (1, 2)]), [], dict(), tahini.core.Edges(index=[(0, 1), (1, 2)])),
    # non empty nodes
    (tahini.core.Edges(), [], dict(nodes=[0, 1]), tahini.core.Edges()),
    # non empty both
    (tahini.core.Edges(index=[(0, 1), (1, 2)]), [], dict(nodes=[0, 1]), tahini.core.Edges(index=[(0, 1)])),
])
def test_edges_keep_nodes(edges, args, kwargs, expected):
    edges = edges.keep_nodes(*args, **kwargs)
    tahini.testing.assert_edges_equal(edges, expected)


@pytest.mark.parametrize('args, kwargs, nodes_expected, edges_expected', [
    # empty
    ([], dict(), tahini.core.Nodes(), tahini.core.Edges()),
    # nodes
    ([], dict(nodes=[0, 1]), tahini.core.Nodes(index=[0, 1]), tahini.core.Edges()),
    # edges
    ([], dict(edges=[(0, 1), (1, 2)]), tahini.core.Nodes(index=[0, 1, 2]), tahini.core.Edges(index=[(0, 1), (1, 2)])),
    # nodes and edges
    (
        [],
        dict(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]),
        tahini.core.Nodes(index=[0, 1, 2]),
        tahini.core.Edges(index=[(0, 1), (1, 2)]),
    ),
    # partial nodes and edges
    (
        [],
        dict(nodes=[0, 1], edges=[(0, 1), (1, 2)]),
        tahini.core.Nodes(index=[0, 1, 2]),
        tahini.core.Edges(index=[(0, 1), (1, 2)]),
    ),
    # order
    ([], dict(order=2), tahini.core.Nodes(index=range(2)), tahini.core.Edges()),
    # nodes data
    (
        [],
        dict(nodes_data=pd.DataFrame(data=dict(value=['a', 'b']))),
        tahini.core.Nodes(data=dict(value=['a', 'b'])),
        tahini.core.Edges(),
    ),
    # edges data
    (
        [],
        dict(edges_data=pd.DataFrame(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)])),
        tahini.core.Nodes(index=[0, 1, 2]),
        tahini.core.Edges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
])
def test_graph_init(args, kwargs, nodes_expected, edges_expected):
    graph = tahini.core.Graph(*args, **kwargs)
    tahini.testing.assert_nodes_equal(graph.nodes, nodes_expected)
    tahini.testing.assert_edges_equal(graph.edges, edges_expected)


@pytest.mark.parametrize('graph, edges, args, kwargs, nodes_expected, graph_expected', [
    # empty
    (
        tahini.core.Graph(),
        tahini.core.Edges(index=[(0, 1)]),
        [],
        dict(),
        tahini.core.Nodes(index=[0, 1]),
        tahini.core.Graph(edges=[(0, 1)]),
    ),
    # non empty
    (
        tahini.core.Graph(nodes=[0]),
        tahini.core.Edges(index=[(1, 2)]),
        [],
        dict(),
        tahini.core.Nodes(index=[0, 1, 2]),
        tahini.core.Graph(nodes=[0, 1, 2], edges=[(1, 2)]),
    ),
])
def test_graph__update_nodes_from_edges(graph, edges, args, kwargs, nodes_expected, graph_expected):
    graph._edges = edges
    nodes = graph._update_nodes_from_edges(*args, **kwargs)
    tahini.testing.assert_nodes_equal(nodes, nodes_expected)
    tahini.testing.assert_graph_equal(graph, graph_expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.Graph(), [], dict(), tahini.core.Graph()),
    # non empty graph
    (tahini.core.Graph(nodes=[0, 1]), [], dict(), tahini.core.Graph(nodes=[0, 1])),
    # non empty nodes
    (tahini.core.Graph(), [], dict(nodes=[0, 1]), tahini.core.Graph(nodes=[0, 1])),
    # non empty both same
    (tahini.core.Graph(nodes=[0, 1]), [], dict(nodes=[0, 1]), tahini.core.Graph(nodes=[0, 1])),
    # non empty both different
    (tahini.core.Graph(nodes=[0, 1]), [], dict(nodes=[2]), tahini.core.Graph(nodes=[0, 1, 2])),
    # data
    (
        tahini.core.Graph(nodes=[0, 1]),
        [],
        dict(data=dict(value=['a', 'b'])),
        tahini.core.Graph(nodes_data=dict(value=['a', 'b']), nodes=[0, 1]),
    ),
])
def test_graph_update_nodes(graph, args, kwargs, expected):
    graph_updated = graph.update_nodes(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph_updated, expected)


@pytest.mark.parametrize('graph, nodes, args, kwargs, edges_expected, graph_expected', [
    # empty
    (
        tahini.core.Graph(),
        tahini.core.Nodes(index=[0, 1]),
        [],
        dict(),
        tahini.core.Edges(),
        tahini.core.Graph(nodes=[0, 1]),
    ),
    # non empty
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        tahini.core.Nodes(index=[0, 1]),
        [],
        dict(),
        tahini.core.Edges(index=[[0, 1]]),
        tahini.core.Graph(edges=[[0, 1]]),
    ),
])
def test_graph__update_edges_from_nodes(graph, nodes, args, kwargs, edges_expected, graph_expected):
    graph._nodes = nodes
    edges = graph._update_edges_from_nodes(*args, **kwargs)
    tahini.testing.assert_edges_equal(edges, edges_expected)
    tahini.testing.assert_graph_equal(graph, graph_expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.Graph(), [], dict(), tahini.core.Graph()),
    # non empty graph
    (tahini.core.Graph(edges=[[0, 1]]), [], dict(), tahini.core.Graph(edges=[[0, 1]])),
    # non empty edges
    (tahini.core.Graph(), [], dict(edges=[[0, 1]]), tahini.core.Graph(edges=[[0, 1]])),
    # non empty both different
    (tahini.core.Graph(edges=[[0, 1]]), [], dict(edges=[[0, 1]]), tahini.core.Graph(edges=[[0, 1]])),
    # non empty both different
    (tahini.core.Graph(edges=[[0, 1]]), [], dict(edges=[[1, 2]]), tahini.core.Graph(edges=[[0, 1], [1, 2]])),
    # data
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(edges=[[0, 1], [1, 2]], data=dict(value=['a', 'b'])),
        tahini.core.Graph(edges_data=dict(value=['a', 'b']), edges=[[0, 1], [1, 2]]),
    ),
])
def test_graph_update_edges(graph, args, kwargs, expected):
    graph_updated = graph.update_edges(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph_updated, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.Graph(), [], dict(), tahini.core.Graph()),
])
def test_graph_update(graph, args, kwargs, expected):
    graph_updated = graph.update(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph_updated, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty all
    (tahini.core.Graph(), [], dict(), tahini.core.Graph()),
    # empty list
    (tahini.core.Graph(), [], dict(nodes=[]), tahini.core.Graph()),
    # empty inputs
    (tahini.core.Graph(nodes=[0, 1]), [], dict(), tahini.core.Graph(nodes=[0, 1])),
    # non empty
    (tahini.core.Graph(nodes=[0, 1]), [], dict(nodes=[0]), tahini.core.Graph(nodes=[1])),
    # non empty with removing edges
    (
        tahini.core.Graph(nodes=[0, 1], edges=[[0, 1]]),
        [],
        dict(nodes=[0]),
        tahini.core.Graph(nodes=[1], edges=tahini.core.Edges()),
    ),
])
def test_graph_drop_nodes(graph, args, kwargs, expected):
    graph_dropped = graph.drop_nodes(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph_dropped, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty all
    (tahini.core.Graph(), [], dict(), tahini.core.Graph()),
    # empty list
    (tahini.core.Graph(), [], dict(edges=[]), tahini.core.Graph()),
    # empty inputs
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(),
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
    ),
    # non empty
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2]]),
        [],
        dict(edges=[(0, 1)]),
        tahini.core.Graph(nodes=[0, 1, 2], edges=[[1, 2]]),
    ),
])
def test_graph_drop_edges(graph, args, kwargs, expected):
    graph_dropped = graph.drop_edges(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph_dropped, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty all
    (tahini.core.Graph(), [], dict(), tahini.core.Graph()),
    # empty lists
    (tahini.core.Graph(), [], dict(nodes=[], edges=[]), tahini.core.Graph()),
    # empty inputs
    (tahini.core.Graph(edges=[[0, 1], [1, 2]]), [], dict(), tahini.core.Graph(edges=[[0, 1], [1, 2]])),
    # non empty
    (
        tahini.core.Graph(edges=[[0, 1], [1, 2], [1, 3]]),
        [],
        dict(nodes=[0], edges=[(1, 2)]),
        tahini.core.Graph(nodes=[1, 2, 3], edges=[[1, 3]]),
    ),
])
def test_graph_drop(graph, args, kwargs, expected):
    graph_dropped = graph.drop(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph_dropped, expected)


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.Graph(), 0),
    # non empty
    (tahini.core.Graph(nodes=[0, 1]), 2),
    # order
    (tahini.core.Graph(order=3), 3),
])
def test_graph_order(graph, expected):
    order = graph.order
    assert order == expected


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.Graph(), 0),
    # non empty nodes
    (tahini.core.Graph(nodes=[0, 1]), 0),
    # order
    (tahini.core.Graph(order=3), 0),
    # non empty edges
    (tahini.core.Graph(edges=[(0, 1)]), 1),
    # non empty edges
    (tahini.core.Graph(edges=[(0, 1), (1, 2)]), 2),
])
def test_graph_size(graph, expected):
    size = graph.size
    assert size == expected


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.Graph(), f'Graph(nodes={tahini.core.Nodes()}, edges={tahini.core.Edges()})'),
])
def test_graph_repr(graph, expected):
    repr_graph = repr(graph)
    assert repr_graph == expected


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.Graph(), [], dict(), pd.Series(dtype='int64', index=pd.Index([], name='node'), name='degree')),
    # non empty
    (
        tahini.core.Graph(edges=[(0, 1)]),
        [],
        dict(),
        pd.Series(data=[1, 1], index=pd.Index([0, 1], name='node'), name='degree'),
    ),
    # non empty with zero degree
    (
        tahini.core.Graph(nodes=[2], edges=[(0, 1)]),
        [],
        dict(),
        pd.Series(data=[1, 1, 0], index=pd.Index([0, 1, 2], name='node'), name='degree'),
    ),
])
def test_graph_get_degree_by_node(graph, args, kwargs, expected):
    degrees = graph.get_degree_by_node(*args, **kwargs)
    pd.testing.assert_series_equal(degrees, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.Graph(), [], dict(), pd.Series(name='neighbors', index=pd.Index([], name='node'))),
    # non empty
    (
        tahini.core.Graph(edges=[(0, 1)]),
        [],
        dict(),
        pd.Series([[1], None], name='neighbors', index=pd.Index([0, 1], name='node')),
    ),
    (
        tahini.core.Graph(nodes=[2], edges=[(0, 1)]),
        [],
        dict(),
        pd.Series([[1], None, None], name='neighbors', index=pd.Index([0, 1, 2], name='node')),
    ),
    (
        tahini.core.Graph(edges=[(0, 1), (1, 0)]),
        [],
        dict(),
        pd.Series([[1], [0]], name='neighbors', index=pd.Index([0, 1], name='node')),
    ),
    (
        tahini.core.Graph(edges=[(0, 1), (1, 2), (0, 2)]),
        [],
        dict(),
        pd.Series([[1, 2], [2], None], name='neighbors', index=pd.Index([0, 1, 2], name='node')),
    ),
])
def test_graph_get_neighbors(graph, args, kwargs, expected):
    neighbors = graph.get_neighbors(*args, **kwargs)
    pd.testing.assert_series_equal(neighbors, expected)

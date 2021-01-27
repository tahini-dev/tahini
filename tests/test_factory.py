import pytest
import pandas as pd

import tahini.factory


def get_data_nodes(name_index='node', *args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(*args, **kwargs).rename_axis(index=name_index)


def get_data_edges(*args, index=None, **kwargs):
    if index is None:
        index = pd.MultiIndex(levels=[[], []], codes=[[], []], names=['node_1', 'node_2'])
    else:
        index = pd.MultiIndex.from_tuples(index, names=['node_1', 'node_2'])
    return pd.DataFrame(*args, index=index, **kwargs)


@pytest.mark.parametrize('args, kwargs, expected_nodes_data, expected_edges_data', [
    # trivial order
    ([], dict(order=0), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(order=1), get_data_nodes(index=range(1)), get_data_edges()),
    # non trivial order
    ([], dict(order=2), get_data_nodes(index=range(2)), get_data_edges(index=[(0, 1)])),
    ([], dict(order=3), get_data_nodes(index=range(3)), get_data_edges(index=[(0, 1), (1, 2)])),
    # order and nodes
    ([], dict(order=2, nodes=['a', 'b']), get_data_nodes(index=['a', 'b']), get_data_edges(index=[('a', 'b')])),
    (
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        get_data_nodes(index=['a', 'b', 'c']),
        get_data_edges(index=[('a', 'b'), ('b', 'c')]),
    ),
    # trivial nodes
    ([], dict(nodes=[]), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(nodes=['a']), get_data_nodes(index=['a']), get_data_edges()),
    # non trivial nodes
    ([], dict(nodes=['a', 'b']), get_data_nodes(index=['a', 'b']), get_data_edges(index=[('a', 'b')])),
])
def test_get_path(args, kwargs, expected_nodes_data, expected_edges_data):
    graph = tahini.factory.get_path(*args, **kwargs)
    pd.testing.assert_frame_equal(graph.nodes.data, expected_nodes_data)
    pd.testing.assert_frame_equal(graph.edges.data, expected_edges_data)


@pytest.mark.parametrize('args, kwargs, expected_nodes_data, expected_edges_data', [
    # trivial order
    ([], dict(order=0), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(order=1), get_data_nodes(index=range(1)), get_data_edges()),
    # non trivial order
    ([], dict(order=2), get_data_nodes(index=range(2)), get_data_edges(index=[(0, 1)])),
    ([], dict(order=3), get_data_nodes(index=range(3)), get_data_edges(index=[(0, 1), (0, 2)])),
    # order and nodes
    ([], dict(order=2, nodes=['a', 'b']), get_data_nodes(index=['a', 'b']), get_data_edges(index=[('a', 'b')])),
    (
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        get_data_nodes(index=['a', 'b', 'c']),
        get_data_edges(index=[('a', 'b'), ('a', 'c')]),
    ),
    # trivial nodes
    ([], dict(nodes=[]), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(nodes=['a']), get_data_nodes(index=['a']), get_data_edges()),
    # non trivial nodes
    ([], dict(nodes=['a', 'b']), get_data_nodes(index=['a', 'b']), get_data_edges(index=[('a', 'b')])),
    (
        [],
        dict(nodes=['a', 'b', 'c']),
        get_data_nodes(index=['a', 'b', 'c']),
        get_data_edges(index=[('a', 'b'), ('a', 'c')]),
    ),
])
def test_get_star(args, kwargs, expected_nodes_data, expected_edges_data):
    graph = tahini.factory.get_star(*args, **kwargs)
    pd.testing.assert_frame_equal(graph.nodes.data, expected_nodes_data)
    pd.testing.assert_frame_equal(graph.edges.data, expected_edges_data)


# todo sort out directed versus undirected graph
@pytest.mark.parametrize('args, kwargs, expected_nodes_data, expected_edges_data', [
    # trivial order
    ([], dict(order=0), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(order=1), get_data_nodes(index=range(1)), get_data_edges()),
    # non trivial order
    ([], dict(order=2), get_data_nodes(index=range(2)), get_data_edges(index=[(0, 1), (1, 0)])),
    (
        [],
        dict(order=3),
        get_data_nodes(index=range(3)),
        get_data_edges(index=[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]),
    ),
    # order and nodes
    (
        [],
        dict(order=2, nodes=['a', 'b']),
        get_data_nodes(index=['a', 'b']),
        get_data_edges(index=[('a', 'b'), ('b', 'a')]),
    ),
    (
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        get_data_nodes(index=['a', 'b', 'c']),
        get_data_edges(index=[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]),
    ),
    # trivial nodes
    ([], dict(nodes=[]), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(nodes=['a']), get_data_nodes(index=['a']), get_data_edges()),
    # non trivial nodes
    ([], dict(nodes=['a', 'b']), get_data_nodes(index=['a', 'b']), get_data_edges(index=[('a', 'b'), ('b', 'a')])),
    (
        [],
        dict(nodes=['a', 'b', 'c']),
        get_data_nodes(index=['a', 'b', 'c']),
        get_data_edges(index=[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]),
    ),
])
def test_get_complete(args, kwargs, expected_nodes_data, expected_edges_data):
    graph = tahini.factory.get_complete(*args, **kwargs)
    pd.testing.assert_frame_equal(graph.nodes.data, expected_nodes_data)
    pd.testing.assert_frame_equal(graph.edges.data, expected_edges_data)

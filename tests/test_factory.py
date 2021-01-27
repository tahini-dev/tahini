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
    ([], dict(order=0), get_data_nodes(index=range(0)), get_data_edges()),
    ([], dict(order=1), get_data_nodes(index=range(1)), get_data_edges()),
    ([], dict(order=2), get_data_nodes(index=range(2)), get_data_edges(index=[(0, 1)])),
    ([], dict(order=3), get_data_nodes(index=range(3)), get_data_edges(index=[(0, 1), (1, 2)])),
    ([], dict(order=2, nodes=['a', 'b']), get_data_nodes(index=['a', 'b']), get_data_edges(index=[('a', 'b')])),
    (
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        get_data_nodes(index=['a', 'b', 'c']),
        get_data_edges(index=[('a', 'b'), ('b', 'c')]),
    ),
])
def test_get_path(args, kwargs, expected_nodes_data, expected_edges_data):
    graph = tahini.factory.get_path(*args, **kwargs)
    pd.testing.assert_frame_equal(graph.nodes.data, expected_nodes_data)
    pd.testing.assert_frame_equal(graph.edges.data, expected_edges_data)

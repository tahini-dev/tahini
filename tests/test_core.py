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


def get_data_frame(name_index, *args, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(*args, **kwargs).rename_axis(index=name_index)


@pytest.mark.parametrize('args, kwargs, expected', [
    # inputs empty
    ([], dict(), get_data_frame(name_index='node')),
    # args - nodes empty
    ([[]], dict(), get_data_frame(name_index='node')),
    # args - data empty
    ([None, []], dict(), get_data_frame(name_index='node')),
    # args - size
    ([None, None, 1], dict(), get_data_frame(name_index='node', index=range(1))),
    # nodes empty
    ([], dict(index=[]), get_data_frame(name_index='node')),
    # data empty
    ([], dict(data=[]), get_data_frame(name_index='node')),
    # empty nodes and data
    ([], dict(index=[], data=[]), get_data_frame(name_index='node')),
    # size zero
    ([], dict(size=0), get_data_frame(name_index='node', index=range(0))),
    # size
    ([], dict(size=1), get_data_frame(name_index='node', index=range(1))),
    # size negative
    ([], dict(size=-1), get_data_frame(name_index='node', index=range(-1))),
    # nodes input it's own class
    ([], dict(index=tahini.core.Nodes()), get_data_frame(name_index='node')),
])
def test_nodes_init(args, kwargs, expected):
    nodes = tahini.core.Nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes.data, expected)
#
#
# @pytest.mark.parametrize('args, kwargs, expected', [
#     # empty
#     ([], dict(), pd.DataFrame()),
#     # edges list of tuples of nodes
#     (
#         [],
#         dict(edges=[(0, 1), (1, 2)]),
#         pd.DataFrame(index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])),
#     ),
#     # edges list of lists of nodes
#     (
#         [],
#         dict(edges=[[0, 1], [1, 2]]),
#         pd.DataFrame(index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])),
#     ),
#     # edges and dict
#     (
#         [],
#         dict(edges=[(0, 1), (1, 2)], data=dict(name=['a', 'b'])),
#         pd.DataFrame(dict(name=['a', 'b']), index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])),
#     ),
#     # data_frame
#     (
#         [],
#         dict(data=pd.DataFrame(dict(name=['a', 'b']), index=[(0, 1), (1, 2)])),
#         pd.DataFrame(dict(name=['a', 'b']), index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])),
#     ),
#     # edges idemponent
#     (
#         [],
#         dict(edges=tahini.core.Edges()),
#         pd.DataFrame(),
#     ),
#     # edges
#     (
#         [],
#         dict(edges=tahini.core.Edges(edges=[(0, 1), (1, 2)])),
#         pd.DataFrame(index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])),
#     ),
#     # edges with values
#     (
#         [],
#         dict(edges=tahini.core.Edges(edges=[(0, 1), (1, 2)], data=dict(name=['a', 'b']))),
#         pd.DataFrame(dict(name=['a', 'b']), index=pd.MultiIndex.from_tuples([(0, 1), (1, 2)])),
#     ),
# ])
# def test_edges_init_simple(args, kwargs, expected):
#     edges = tahini.core.Edges(*args, **kwargs)
#     pd.testing.assert_frame_equal(edges.data, expected)
#
# # @pytest.mark.parametrize('args, kwargs', [
# #     ([], dict()),
# # ])
# # def test_graph_init(args, kwargs):
# #     graph = tahini.core.Graph(*args, **kwargs)
# #     assert isinstance(graph, tahini.core.Graph)

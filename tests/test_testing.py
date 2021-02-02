import pytest

import tahini.testing
from tahini.core import Nodes, Edges, Graph


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # different length
    (
        [],
        dict(left=Nodes(), right=Nodes(order=1)),
        AssertionError,
        '''Nodes.data are different

Nodes.data shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # different length
    (
        [],
        dict(left=Nodes(), right=Nodes(index=[1])),
        AssertionError,
        '''Nodes.data are different

Nodes.data shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # different index
    (
        [],
        dict(left=Nodes(index=[1]), right=Nodes(index=[2])),
        AssertionError,
        '''Nodes.data.index are different

Nodes.data.index values are different (100.0 %)
[left]:  Int64Index([1], dtype='int64', name='node')
[right]: Int64Index([2], dtype='int64', name='node')''',
    ),
    # different type
    (
        [],
        dict(left=Nodes(index=[1]), right=Nodes(index=['a'])),
        AssertionError,
        '''Nodes.data.index are different

Nodes.data.index values are different (100.0 %)
[left]:  Int64Index([1], dtype='int64', name='node')
[right]: Index(['a'], dtype='object', name='node')''',
    ),
    # different order
    (
        [],
        dict(left=Nodes(index=[0, 1]), right=Nodes(index=[1, 0])),
        AssertionError,
        '''Nodes.data.index are different

Nodes.data.index values are different (100.0 %)
[left]:  Int64Index([0, 1], dtype='int64', name='node')
[right]: Int64Index([1, 0], dtype='int64', name='node')''',
    ),
])
def test_assert_nodes_equal_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.testing.assert_nodes_equal(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict(left=Nodes(), right=Nodes())),
    # non empty
    ([], dict(left=Nodes(order=1), right=Nodes(order=1))),
    # range versus int type
    ([], dict(left=Nodes(index=range(1)), right=Nodes(index=[0]))),
    # dropped
    ([], dict(left=Nodes(index=range(1)).drop(index=[0]), right=Nodes())),
])
def test_assert_nodes_equal(args, kwargs):
    tahini.testing.assert_nodes_equal(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # different length
    (
        [],
        dict(left=Edges(), right=Edges(index=[(0, 1)])),
        AssertionError,
        '''Edges.data are different

Edges.data shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # different index
    (
        [],
        dict(left=Edges(index=[(0, 1)]), right=Edges(index=[(0, 2)])),
        AssertionError,
        '''Edges.data.index node [1] are different

Edges.data.index node [1] values are different (100.0 %)
[left]:  Int64Index([1], dtype='int64', name='node_to')
[right]: Int64Index([2], dtype='int64', name='node_to')''',
    ),
    # different type
    (
        [],
        dict(left=Edges(index=[(0, 1)]), right=Edges(index=[('a', 'b')])),
        AssertionError,
        '''Edges.data.index node [0] are different

Edges.data.index node [0] values are different (100.0 %)
[left]:  Int64Index([0], dtype='int64', name='node_from')
[right]: Index(['a'], dtype='object', name='node_from')''',
    ),
    # different order
    (
        [],
        dict(left=Edges(index=[(0, 1), (0, 2)]), right=Edges(index=[(0, 2), (0, 1)])),
        AssertionError,
        '''Edges.data.index node [1] are different

Edges.data.index node [1] values are different (100.0 %)
[left]:  Int64Index([1, 2], dtype='int64', name='node_to')
[right]: Int64Index([2, 1], dtype='int64', name='node_to')''',
    ),
])
def test_assert_edges_equal_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.testing.assert_edges_equal(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict(left=Edges(), right=Edges())),
    # non empty
    ([], dict(left=Edges(index=[(0, 1)]), right=Edges(index=[(0, 1)]))),
    # dropped
    ([], dict(left=Edges(index=[(0, 1)]).drop(index=[(0, 1)]), right=Edges())),
])
def test_assert_edges_equal(args, kwargs):
    tahini.testing.assert_edges_equal(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # different length nodes
    (
        [],
        dict(left=Graph(), right=Graph(order=1)),
        AssertionError,
        '''Graph.nodes.data are different

Graph.nodes.data shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # different length edges
    (
        [],
        dict(left=Graph(nodes=[0, 1]), right=Graph(edges=[(0, 1)])),
        AssertionError,
        '''Graph.edges.data are different

Graph.edges.data shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # non default obj
    (
        [],
        dict(left=Graph(nodes=[0, 1]), right=Graph(edges=[(0, 1)]), obj='ChildGraph'),
        AssertionError,
        '''ChildGraph are different

ChildGraph shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
])
def test_assert_graph_equal_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.testing.assert_graph_equal(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict(left=Graph(), right=Graph())),
    # obj
    ([], dict(left=Graph(), right=Graph(), obj='ChildGraph')),
    # non empty nodes
    ([], dict(left=Graph(order=1), right=Graph(order=1))),
    ([], dict(left=Graph(order=2), right=Graph(order=2))),
    # non empty edges
    ([], dict(left=Graph(edges=[(0, 1)]), right=Graph(edges=[(0, 1)]))),
    # non empty edges with nodes extra
    ([], dict(left=Graph(edges=[(0, 1), (1, 2)]), right=Graph(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]))),
])
def test_assert_graph_equal(args, kwargs):
    tahini.testing.assert_graph_equal(*args, **kwargs)

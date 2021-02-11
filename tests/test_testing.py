import pytest

import tahini.testing
import tahini.container
import tahini.core


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # mismatch in types
    (
        [],
        dict(left=tahini.container.ContainerDataIndexed(), right=tahini.container.ContainerDataIndexedMulti()),
        AssertionError,
        """Types are different

[left]:  ContainerDataIndexed
[right]: ContainerDataIndexedMulti""",
    ),
    (
        [],
        dict(left=tahini.container.ContainerDataIndexedMulti(), right=tahini.container.ContainerDataIndexedMultiSets()),
        AssertionError,
        """Types are different

[left]:  ContainerDataIndexedMulti
[right]: ContainerDataIndexedMultiSets""",
    ),
    # different length
    (
        [],
        dict(left=tahini.container.ContainerDataIndexed(), right=tahini.container.ContainerDataIndexed(index=[0])),
        AssertionError,
        """ContainerDataIndexed.data_testing are different

ContainerDataIndexed.data_testing shape mismatch
[left]:  (0, 0)
[right]: (1, 0)""",
    ),
    # different index
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexed(index=[0]),
            right=tahini.container.ContainerDataIndexed(index=[1]),
        ),
        AssertionError,
        """ContainerDataIndexed.data_testing.index are different

ContainerDataIndexed.data_testing.index values are different (100.0 %)
[left]:  Int64Index([0], dtype='int64', name='index_internal')
[right]: Int64Index([1], dtype='int64', name='index_internal')""",
    ),
    # different index type
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexed(index=[0]),
            right=tahini.container.ContainerDataIndexed(index=['a']),
        ),
        AssertionError,
        """ContainerDataIndexed.data_testing.index are different

ContainerDataIndexed.data_testing.index values are different (100.0 %)
[left]:  Int64Index([0], dtype='int64', name='index_internal')
[right]: Index(['a'], dtype='object', name='index_internal')""",
    ),
    # different order
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
            right=tahini.container.ContainerDataIndexedMulti(index=[(1, 0)]),
        ),
        AssertionError,
        """ContainerDataIndexedMulti.data_testing.index are different

ContainerDataIndexedMulti.data_testing.index values are different (100.0 %)
[left]:  Index([(0, 1)], dtype='object', name='index_internal')
[right]: Index([(1, 0)], dtype='object', name='index_internal')""",
    ),
])
def test_assert_container_equal_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.testing.assert_container_equal(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict(left=tahini.container.ContainerDataIndexed(), right=tahini.container.ContainerDataIndexed())),
    ([], dict(left=tahini.container.ContainerDataIndexedMulti(), right=tahini.container.ContainerDataIndexedMulti())),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMultiSets(),
            right=tahini.container.ContainerDataIndexedMultiSets()),
    ),
    # non empty
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexed(index=[0]),
            right=tahini.container.ContainerDataIndexed(index=[0])),
    ),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexed(index=[0, 1]),
            right=tahini.container.ContainerDataIndexed(index=[0, 1])),
    ),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexed(index=[0, 1]),
            right=tahini.container.ContainerDataIndexed(index=[1, 0])),
    ),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
            right=tahini.container.ContainerDataIndexedMulti(index=[(0, 1)])),
    ),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]),
            right=tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)])),
    ),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
            right=tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)])),
    ),
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]),
            right=tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)])),
    ),
    # order does not matter for ContainerDataIndexedMultiSets
    (
        [],
        dict(
            left=tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
            right=tahini.container.ContainerDataIndexedMultiSets(index=[(1, 0)])),
    ),
])
def test_assert_container_equal(args, kwargs):
    tahini.testing.assert_container_equal(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # different type
    (
        [],
        dict(left=tahini.core.Graph(), right=tahini.core.UndirectedGraph()),
        AssertionError,
        '''Types are different

[left]:  Edges
[right]: UndirectedEdges''',
    ),
    # different length nodes
    (
        [],
        dict(left=tahini.core.Graph(), right=tahini.core.Graph(order=1)),
        AssertionError,
        '''Graph.nodes.data_testing are different

Graph.nodes.data_testing shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # different length nodes for undirected
    (
        [],
        dict(left=tahini.core.UndirectedGraph(), right=tahini.core.UndirectedGraph(order=1)),
        AssertionError,
        '''UndirectedGraph.nodes.data_testing are different

UndirectedGraph.nodes.data_testing shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # different length edges
    (
        [],
        dict(left=tahini.core.Graph(nodes=[0, 1]), right=tahini.core.Graph(edges=[(0, 1)])),
        AssertionError,
        '''Graph.edges.data_testing are different

Graph.edges.data_testing shape mismatch
[left]:  (0, 0)
[right]: (1, 0)''',
    ),
    # order of each edge matters for graph
    (
        [],
        dict(
            left=tahini.core.Graph(edges=[(0, 1)]),
            right=tahini.core.Graph(edges=[(1, 0)]),
        ),
        AssertionError,
        '''Graph.edges.data_testing.index are different

Graph.edges.data_testing.index values are different (100.0 %)
[left]:  Index([(0, 1)], dtype='object', name='edge_internal')
[right]: Index([(1, 0)], dtype='object', name='edge_internal')''',
    ),
    # non default obj
    (
        [],
        dict(left=tahini.core.Graph(nodes=[0, 1]), right=tahini.core.Graph(edges=[(0, 1)]), obj='ChildGraph'),
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
    ([], dict(left=tahini.core.Graph(), right=tahini.core.Graph())),
    ([], dict(left=tahini.core.UndirectedGraph(), right=tahini.core.UndirectedGraph())),
    # obj
    ([], dict(left=tahini.core.Graph(), right=tahini.core.Graph(), obj='ChildGraph')),
    ([], dict(left=tahini.core.UndirectedGraph(), right=tahini.core.UndirectedGraph(), obj='ChildGraph')),
    # non empty nodes
    ([], dict(left=tahini.core.Graph(order=1), right=tahini.core.Graph(order=1))),
    ([], dict(left=tahini.core.Graph(order=2), right=tahini.core.Graph(order=2))),
    ([], dict(left=tahini.core.UndirectedGraph(order=1), right=tahini.core.UndirectedGraph(order=1))),
    ([], dict(left=tahini.core.UndirectedGraph(order=2), right=tahini.core.UndirectedGraph(order=2))),
    # non empty edges
    ([], dict(left=tahini.core.Graph(edges=[(0, 1)]), right=tahini.core.Graph(edges=[(0, 1)]))),
    ([], dict(left=tahini.core.UndirectedGraph(edges=[(0, 1)]), right=tahini.core.UndirectedGraph(edges=[(0, 1)]))),
    # non empty edges with nodes extra
    (
        [],
        dict(
            left=tahini.core.Graph(edges=[(0, 1), (1, 2)]),
            right=tahini.core.Graph(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]),
        ),
    ),
    (
        [],
        dict(
            left=tahini.core.UndirectedGraph(edges=[(0, 1), (1, 2)]),
            right=tahini.core.UndirectedGraph(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]),
        ),
    ),
    # order does not matter for undirected graph
    (
        [],
        dict(
            left=tahini.core.UndirectedGraph(edges=[(0, 1)]),
            right=tahini.core.UndirectedGraph(edges=[(1, 0)]),
        ),
    ),
])
def test_assert_graph_equal(args, kwargs):
    tahini.testing.assert_graph_equal(*args, **kwargs)

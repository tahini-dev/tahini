from functools import partial

import pytest
import pandas as pd

import tahini.core.graph
import tahini.core.edges
import tahini.core.nodes
import tahini.testing


names_edges = tahini.core.graph.Graph().edges.names_index
name_nodes = tahini.core.graph.Graph().nodes.names_index[0]

assert_frame_equal = partial(
    pd.testing.assert_frame_equal,
    check_column_type=False,
    check_index_type=False,
    check_dtype=False,
)


@pytest.mark.parametrize('args, kwargs, nodes_expected, edges_expected', [
    # empty
    ([], dict(), tahini.core.nodes.Nodes(), tahini.core.edges.Edges()),
    # nodes
    ([], dict(nodes=[0, 1]), tahini.core.nodes.Nodes(index=[0, 1]), tahini.core.edges.Edges()),
    # edges
    (
        [],
        dict(edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.Edges(index=[(0, 1), (1, 2)]),
    ),
    # nodes and edges
    (
        [],
        dict(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.Edges(index=[(0, 1), (1, 2)]),
    ),
    # partial nodes and edges
    (
        [],
        dict(nodes=[0, 1], edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.Edges(index=[(0, 1), (1, 2)]),
    ),
    # order
    ([], dict(order=2), tahini.core.nodes.Nodes(index=range(2)), tahini.core.edges.Edges()),
    # nodes data
    (
        [],
        dict(nodes_data=pd.DataFrame(data=dict(value=['a', 'b']))),
        tahini.core.nodes.Nodes(data=dict(value=['a', 'b'])),
        tahini.core.edges.Edges(),
    ),
    # edges data
    (
        [],
        dict(edges_data=pd.DataFrame(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)])),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.Edges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
])
def test_graph_init(args, kwargs, nodes_expected, edges_expected):
    graph = tahini.core.graph.Graph(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(graph.nodes, nodes_expected)
    tahini.testing.testing.assert_container_equal(graph.edges, edges_expected)


@pytest.mark.parametrize('args, kwargs, nodes_expected, edges_expected', [
    # empty
    ([], dict(), tahini.core.nodes.Nodes(), tahini.core.edges.UndirectedEdges()),
    # nodes
    ([], dict(nodes=[0, 1]), tahini.core.nodes.Nodes(index=[0, 1]), tahini.core.edges.UndirectedEdges()),
    # edges
    (
        [],
        dict(edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1), (1, 2)]),
    ),
    # nodes and edges
    (
        [],
        dict(nodes=[0, 1, 2], edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1), (1, 2)]),
    ),
    # partial nodes and edges
    (
        [],
        dict(nodes=[0, 1], edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1), (1, 2)]),
    ),
    # order
    ([], dict(order=2), tahini.core.nodes.Nodes(index=range(2)), tahini.core.edges.UndirectedEdges()),
    # nodes data
    (
        [],
        dict(nodes_data=pd.DataFrame(data=dict(value=['a', 'b']))),
        tahini.core.nodes.Nodes(data=dict(value=['a', 'b'])),
        tahini.core.edges.UndirectedEdges(),
    ),
    # edges data
    (
        [],
        dict(edges_data=pd.DataFrame(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)])),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.edges.UndirectedEdges(data=dict(value=['a', 'b']), index=[(0, 1), (1, 2)]),
    ),
])
def test_undirected_graph_init(args, kwargs, nodes_expected, edges_expected):
    graph = tahini.core.graph.UndirectedGraph(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(graph.nodes, nodes_expected)
    tahini.testing.testing.assert_container_equal(graph.edges, edges_expected)


@pytest.mark.parametrize('klass, attribute, expected', [
    # _type_edges
    (tahini.core.graph.Graph, '_type_edges', tahini.core.edges.Edges),
    (tahini.core.graph.UndirectedGraph, '_type_edges', tahini.core.edges.UndirectedEdges),
    # _columns_degree_keep
    (tahini.core.graph.Graph, '_columns_degree_keep', ['degree_in', 'degree_out', 'degree']),
    (tahini.core.graph.UndirectedGraph, '_columns_degree_keep', ['degree']),
    # _columns_neighbors_keep
    (tahini.core.graph.Graph, '_columns_neighbors_keep', ['neighbors_in', 'neighbors_out', 'neighbors']),
    (tahini.core.graph.UndirectedGraph, '_columns_neighbors_keep', ['neighbors']),
])
def test_graph_attribute(klass, attribute, expected):
    value = getattr(klass, attribute)
    assert value == expected


@pytest.mark.parametrize('klass, args, kwargs, expected', [
    # trivial order
    (tahini.core.graph.Graph, [], dict(order=0), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph, [], dict(order=1), tahini.core.graph.Graph(order=1)),
    (tahini.core.graph.UndirectedGraph, [], dict(order=0), tahini.core.graph.UndirectedGraph()),
    (tahini.core.graph.UndirectedGraph, [], dict(order=1), tahini.core.graph.UndirectedGraph(order=1)),
    # non trivial order
    (tahini.core.graph.Graph, [], dict(order=2), tahini.core.graph.Graph(edges=[(0, 1)])),
    (tahini.core.graph.Graph, [], dict(order=3), tahini.core.graph.Graph(edges=[(0, 1), (1, 2)])),
    (tahini.core.graph.UndirectedGraph, [], dict(order=2), tahini.core.graph.UndirectedGraph(edges=[(0, 1)])),
    (tahini.core.graph.UndirectedGraph, [], dict(order=3), tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)])),
    # order and nodes
    (tahini.core.graph.Graph, [], dict(order=2, nodes=['a', 'b']), tahini.core.graph.Graph(edges=[('a', 'b')])),
    (
        tahini.core.graph.Graph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('b', 'c')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=2, nodes=['a', 'b']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('b', 'c')]),
    ),
    # trivial nodes
    (tahini.core.graph.Graph, [], dict(nodes=[]), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph, [], dict(nodes=['a']), tahini.core.graph.Graph(nodes=['a'])),
    (tahini.core.graph.UndirectedGraph, [], dict(nodes=[]), tahini.core.graph.UndirectedGraph()),
    (tahini.core.graph.UndirectedGraph, [], dict(nodes=['a']), tahini.core.graph.UndirectedGraph(nodes=['a'])),
    # non trivial nodes
    (tahini.core.graph.Graph, [], dict(nodes=['a', 'b']), tahini.core.graph.Graph(edges=[('a', 'b')])),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(nodes=['a', 'b']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
])
def test_graph_path(klass, args, kwargs, expected):
    graph = klass.path(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph, expected)


@pytest.mark.parametrize('klass, args, kwargs, type_error, message_error', [
    # trivial order
    (
        tahini.core.graph.Graph,
        [],
        dict(order=0),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.Graph,
        [],
        dict(order=1),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.Graph,
        [],
        dict(order=2),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=0),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=1),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=2),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    # trivial nodes
    (
        tahini.core.graph.Graph,
        [], dict(nodes=[]),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.Graph,
        [],
        dict(nodes=[0]),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
        tahini.core.graph.Graph,
        [],
        dict(nodes=[0, 1]),
        ValueError,
        "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
            tahini.core.graph.UndirectedGraph,
            [],
            dict(nodes=[]),
            ValueError,
            "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
            tahini.core.graph.UndirectedGraph,
            [],
            dict(nodes=[0]),
            ValueError,
            "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
    (
            tahini.core.graph.UndirectedGraph,
            [],
            dict(nodes=[0, 1]),
            ValueError,
            "Inputs 'order' or length of 'nodes' has to be >= 3 for cycle",
    ),
])
def test_graph_cycle_error(klass, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        klass.cycle(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('klass, args, kwargs, expected', [
    # non trivial order
    (tahini.core.graph.Graph, [], dict(order=3), tahini.core.graph.Graph(edges=[(0, 1), (1, 2), (2, 0)])),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=3),
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2), (2, 0)]),
    ),
    # order and nodes
    (
        tahini.core.graph.Graph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('b', 'c'), ('c', 'a')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('b', 'c'), ('c', 'a')]),
    ),
    # non trivial nodes
    (
        tahini.core.graph.Graph,
        [],
        dict(nodes=['a', 'b', 'c']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('b', 'c'), ('c', 'a')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('b', 'c'), ('c', 'a')]),
    ),
])
def test_graph_cycle(klass, args, kwargs, expected):
    graph = klass.cycle(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph, expected)


@pytest.mark.parametrize('klass, args, kwargs, expected', [
    # trivial order
    (tahini.core.graph.Graph, [], dict(order=0), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph, [], dict(order=1), tahini.core.graph.Graph(order=1)),
    (tahini.core.graph.UndirectedGraph, [], dict(order=0), tahini.core.graph.UndirectedGraph()),
    (tahini.core.graph.UndirectedGraph, [], dict(order=1), tahini.core.graph.UndirectedGraph(order=1)),
    # non trivial order
    (tahini.core.graph.Graph, [], dict(order=2), tahini.core.graph.Graph(edges=[(0, 1)])),
    (tahini.core.graph.Graph, [], dict(order=3), tahini.core.graph.Graph(edges=[(0, 1), (0, 2)])),
    (tahini.core.graph.UndirectedGraph, [], dict(order=2), tahini.core.graph.UndirectedGraph(edges=[(0, 1)])),
    (tahini.core.graph.UndirectedGraph, [], dict(order=3), tahini.core.graph.UndirectedGraph(edges=[(0, 1), (0, 2)])),
    # order and nodes
    (tahini.core.graph.Graph, [], dict(order=2, nodes=['a', 'b']), tahini.core.graph.Graph(edges=[('a', 'b')])),
    (
        tahini.core.graph.Graph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('a', 'c')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=2, nodes=['a', 'b']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('a', 'c')]),
    ),
    # trivial nodes
    (tahini.core.graph.Graph, [], dict(nodes=[]), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph, [], dict(nodes=['a']), tahini.core.graph.Graph(nodes=['a'])),
    (tahini.core.graph.UndirectedGraph, [], dict(nodes=[]), tahini.core.graph.UndirectedGraph()),
    (tahini.core.graph.UndirectedGraph, [], dict(nodes=['a']), tahini.core.graph.UndirectedGraph(nodes=['a'])),
    # non trivial nodes
    (tahini.core.graph.Graph, [], dict(nodes=['a', 'b']), tahini.core.graph.Graph(edges=[('a', 'b')])),
    (tahini.core.graph.Graph, [], dict(nodes=['a', 'b', 'c']), tahini.core.graph.Graph(edges=[('a', 'b'), ('a', 'c')])),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(nodes=['a', 'b']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('a', 'c')]),
    ),
])
def test_graph_star(klass, args, kwargs, expected):
    graph = klass.star(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph, expected)


@pytest.mark.parametrize('klass, args, kwargs, expected', [
    # trivial order
    (tahini.core.graph.Graph, [], dict(order=0), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph, [], dict(order=1), tahini.core.graph.Graph(order=1)),
    (tahini.core.graph.UndirectedGraph, [], dict(order=0), tahini.core.graph.UndirectedGraph()),
    (tahini.core.graph.UndirectedGraph, [], dict(order=1), tahini.core.graph.UndirectedGraph(order=1)),
    # non trivial order
    (tahini.core.graph.Graph, [], dict(order=2), tahini.core.graph.Graph(edges=[(0, 1), (1, 0)])),
    (
        tahini.core.graph.Graph,
        [], dict(order=3),
        tahini.core.graph.Graph(edges=[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]),
    ),
    (tahini.core.graph.UndirectedGraph, [], dict(order=2), tahini.core.graph.UndirectedGraph(edges=[(0, 1)])),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=3),
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (0, 2), (1, 2)]),
    ),
    # order and nodes
    (
        tahini.core.graph.Graph,
        [],
        dict(order=2, nodes=['a', 'b']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('b', 'a')]),
    ),
    (
        tahini.core.graph.Graph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=2, nodes=['a', 'b']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('a', 'c'), ('b', 'c')]),
    ),
    # trivial nodes
    (tahini.core.graph.Graph, [], dict(nodes=[]), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph, [], dict(nodes=['a']), tahini.core.graph.Graph(nodes=['a'])),
    (tahini.core.graph.UndirectedGraph, [], dict(nodes=[]), tahini.core.graph.UndirectedGraph()),
    (tahini.core.graph.UndirectedGraph, [], dict(nodes=['a']), tahini.core.graph.UndirectedGraph(nodes=['a'])),
    # non trivial nodes
    (tahini.core.graph.Graph, [], dict(nodes=['a', 'b']), tahini.core.graph.Graph(edges=[('a', 'b'), ('b', 'a')])),
    (
        tahini.core.graph.Graph,
        [],
        dict(nodes=['a', 'b', 'c']),
        tahini.core.graph.Graph(edges=[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(nodes=['a', 'b']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph,
        [],
        dict(nodes=['a', 'b', 'c']),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b'), ('a', 'c'), ('b', 'c')]),
    ),
])
def test_graph_complete(klass, args, kwargs, expected):
    graph = klass.complete(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.graph.Graph(), [], dict(), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph(), [], dict(nodes=lambda x: x.nodes), tahini.core.graph.Graph()),
    (tahini.core.graph.Graph(), [], dict(nodes=lambda x: x.edges), tahini.core.graph.Graph()),
    # non empty inputs
    (tahini.core.graph.Graph(), [], dict(nodes=[0]), tahini.core.graph.Graph(nodes=[0])),
    (tahini.core.graph.Graph(), [], dict(edges=[(0, 1)]), tahini.core.graph.Graph(edges=[(0, 1)])),
    # non empty graph
    (tahini.core.graph.Graph(nodes=[0]), [], dict(nodes=[1]), tahini.core.graph.Graph(nodes=[1])),
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        [],
        dict(edges=[(0, 2)]),
        tahini.core.graph.Graph(nodes=[1], edges=[(0, 2)]),
    ),
])
def test_graph_assign(graph, args, kwargs, expected):
    graph_assigned = graph.assign(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph_assigned, expected)


@pytest.mark.parametrize('graph, edges, args, kwargs, nodes_expected, graph_expected', [
    # empty
    (
        tahini.core.graph.Graph(),
        tahini.core.edges.Edges(index=[(0, 1)]),
        [],
        dict(),
        tahini.core.nodes.Nodes(index=[0, 1]),
        tahini.core.graph.Graph(edges=[(0, 1)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(),
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
        [],
        dict(),
        tahini.core.nodes.Nodes(index=[0, 1]),
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
    ),
    # non empty
    (
        tahini.core.graph.Graph(nodes=[0]),
        tahini.core.edges.Edges(index=[(1, 2)]),
        [],
        dict(),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.graph.Graph(nodes=[0, 1, 2], edges=[(1, 2)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0]),
        tahini.core.edges.UndirectedEdges(index=[(1, 2)]),
        [],
        dict(),
        tahini.core.nodes.Nodes(index=[0, 1, 2]),
        tahini.core.graph.UndirectedGraph(nodes=[0, 1, 2], edges=[(1, 2)]),
    ),
])
def test_graph__update_nodes_from_edges(graph, edges, args, kwargs, nodes_expected, graph_expected):
    graph._edges = edges
    nodes = graph._update_nodes_from_edges(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(nodes, nodes_expected)
    tahini.testing.testing.assert_graph_equal(graph, graph_expected)


@pytest.mark.parametrize('graph, args, kwargs, type_error, message_error', [
    # non unique update
    (tahini.core.graph.Graph(), [], dict(nodes=[0, 0]), ValueError, "Index needs to be unique for 'Nodes'"),
])
def test_graph_update_nodes_error(graph, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        graph.update_nodes(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.graph.Graph(), [], dict(), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(), tahini.core.graph.UndirectedGraph()),
    # non empty graph
    (tahini.core.graph.Graph(nodes=[0, 1]), [], dict(), tahini.core.graph.Graph(nodes=[0, 1])),
    (tahini.core.graph.UndirectedGraph(nodes=[0, 1]), [], dict(), tahini.core.graph.UndirectedGraph(nodes=[0, 1])),
    # non empty nodes
    (tahini.core.graph.Graph(), [], dict(nodes=[0, 1]), tahini.core.graph.Graph(nodes=[0, 1])),
    (tahini.core.graph.UndirectedGraph(), [], dict(nodes=[0, 1]), tahini.core.graph.UndirectedGraph(nodes=[0, 1])),
    # non empty both same
    (tahini.core.graph.Graph(nodes=[0, 1]), [], dict(nodes=[0, 1]), tahini.core.graph.Graph(nodes=[0, 1])),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
        [],
        dict(nodes=[0, 1]),
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
    ),
    # non empty both different
    (tahini.core.graph.Graph(nodes=[0, 1]), [], dict(nodes=[2]), tahini.core.graph.Graph(nodes=[0, 1, 2])),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
        [],
        dict(nodes=[2]),
        tahini.core.graph.UndirectedGraph(nodes=[0, 1, 2]),
    ),
    # data
    (
        tahini.core.graph.Graph(nodes=[0, 1]),
        [],
        dict(data=dict(value=['a', 'b'])),
        tahini.core.graph.Graph(nodes_data=dict(value=['a', 'b']), nodes=[0, 1]),
    ),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
        [],
        dict(data=dict(value=['a', 'b'])),
        tahini.core.graph.UndirectedGraph(nodes_data=dict(value=['a', 'b']), nodes=[0, 1]),
    ),
])
def test_graph_update_nodes(graph, args, kwargs, expected):
    graph_updated = graph.update_nodes(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph_updated, expected)


@pytest.mark.parametrize('graph, nodes, args, kwargs, edges_expected, graph_expected', [
    # empty
    (
        tahini.core.graph.Graph(),
        tahini.core.nodes.Nodes(index=[0, 1]),
        [],
        dict(),
        tahini.core.edges.Edges(),
        tahini.core.graph.Graph(nodes=[0, 1]),
    ),
    (
        tahini.core.graph.UndirectedGraph(),
        tahini.core.nodes.Nodes(index=[0, 1]),
        [],
        dict(),
        tahini.core.edges.UndirectedEdges(),
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
    ),
    # non empty
    (
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1]),
        [],
        dict(),
        tahini.core.edges.Edges(index=[(0, 1)]),
        tahini.core.graph.Graph(edges=[(0, 1)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)]),
        tahini.core.nodes.Nodes(index=[0, 1]),
        [],
        dict(),
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
    ),
])
def test_graph__update_edges_from_nodes(graph, nodes, args, kwargs, edges_expected, graph_expected):
    graph._nodes = nodes
    edges = graph._update_edges_from_nodes(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(edges, edges_expected)
    tahini.testing.testing.assert_graph_equal(graph, graph_expected)


@pytest.mark.parametrize('graph, args, kwargs, type_error, message_error', [
    # non unique update
    (tahini.core.graph.Graph(), [], dict(edges=[(0, 1), (0, 1)]), ValueError, "Index needs to be unique for 'Edges'"),
    (
        tahini.core.graph.UndirectedGraph(),
        [],
        dict(edges=[(0, 1), (1, 0)]),
        ValueError,
        "Index needs to be unique for 'UndirectedEdges'",
    ),
])
def test_graph_update_edges_error(graph, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        graph.update_edges(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.graph.Graph(), [], dict(), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(), tahini.core.graph.UndirectedGraph()),
    # non empty graph
    (tahini.core.graph.Graph(edges=[(0, 1)]), [], dict(), tahini.core.graph.Graph(edges=[(0, 1)])),
    (tahini.core.graph.UndirectedGraph(edges=[(0, 1)]), [], dict(), tahini.core.graph.UndirectedGraph(edges=[(0, 1)])),
    # non empty edges
    (tahini.core.graph.Graph(), [], dict(edges=[(0, 1)]), tahini.core.graph.Graph(edges=[(0, 1)])),
    (tahini.core.graph.UndirectedGraph(), [], dict(edges=[(0, 1)]), tahini.core.graph.UndirectedGraph(edges=[(0, 1)])),
    # non empty both different
    (tahini.core.graph.Graph(edges=[(0, 1)]), [], dict(edges=[(0, 1)]), tahini.core.graph.Graph(edges=[(0, 1)])),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
        [],
        dict(edges=[(0, 1)]),
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
    ),
    # non empty both different
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        [],
        dict(edges=[(1, 2)]),
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
        [],
        dict(edges=[(1, 2)]),
        tahini.core.graph.UndirectedGraph(edges=[(1, 2), (0, 1)]),
    ),
    # data
    (
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]),
        [],
        dict(edges=[(0, 1), (1, 2)], data=dict(value=['a', 'b'])),
        tahini.core.graph.Graph(edges_data=dict(value=['a', 'b']), edges=[(0, 1), (1, 2)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)]),
        [],
        dict(edges=[(0, 1), (1, 2)], data=dict(value=['a', 'b'])),
        tahini.core.graph.UndirectedGraph(edges_data=dict(value=['a', 'b']), edges=[(0, 1), (1, 2)]),
    ),
])
def test_graph_update_edges(graph, args, kwargs, expected):
    graph_updated = graph.update_edges(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph_updated, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty all
    (tahini.core.graph.Graph(), [], dict(), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(), tahini.core.graph.UndirectedGraph()),
    # empty list
    (tahini.core.graph.Graph(), [], dict(nodes=[]), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(nodes=[]), tahini.core.graph.UndirectedGraph()),
    # empty inputs
    (tahini.core.graph.Graph(nodes=[0, 1]), [], dict(), tahini.core.graph.Graph(nodes=[0, 1])),
    (tahini.core.graph.UndirectedGraph(nodes=[0, 1]), [], dict(), tahini.core.graph.UndirectedGraph(nodes=[0, 1])),
    # non empty
    (tahini.core.graph.Graph(nodes=[0, 1]), [], dict(nodes=[0]), tahini.core.graph.Graph(nodes=[1])),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
        [],
        dict(nodes=[0]),
        tahini.core.graph.UndirectedGraph(nodes=[1]),
    ),
    # non empty with removing edges
    (
        tahini.core.graph.Graph(nodes=[0, 1], edges=[(0, 1)]),
        [],
        dict(nodes=[0]),
        tahini.core.graph.Graph(nodes=[1], edges=tahini.core.edges.Edges()),
    ),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0, 1], edges=[(0, 1)]),
        [],
        dict(nodes=[0]),
        tahini.core.graph.UndirectedGraph(nodes=[1], edges=tahini.core.edges.Edges()),
    ),
])
def test_graph_drop_nodes(graph, args, kwargs, expected):
    graph_dropped = graph.drop_nodes(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph_dropped, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty all
    (tahini.core.graph.Graph(), [], dict(), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(), tahini.core.graph.UndirectedGraph()),
    # empty list
    (tahini.core.graph.Graph(), [], dict(edges=[]), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(edges=[]), tahini.core.graph.UndirectedGraph()),
    # empty inputs
    (
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]),
        [],
        dict(),
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)]),
        [],
        dict(),
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)]),
    ),
    # non empty
    (
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]),
        [],
        dict(edges=[(0, 1)]),
        tahini.core.graph.Graph(nodes=[0, 1, 2], edges=[(1, 2)]),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)]),
        [],
        dict(edges=[(0, 1)]),
        tahini.core.graph.UndirectedGraph(nodes=[0, 1, 2], edges=[(1, 2)]),
    ),
])
def test_graph_drop_edges(graph, args, kwargs, expected):
    graph_dropped = graph.drop_edges(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph_dropped, expected)


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.graph.Graph(), 0),
    (tahini.core.graph.UndirectedGraph(), 0),
    # non empty
    (tahini.core.graph.Graph(nodes=[0, 1]), 2),
    (tahini.core.graph.UndirectedGraph(nodes=[0, 1]), 2),
    # order
    (tahini.core.graph.Graph(order=3), 3),
    (tahini.core.graph.UndirectedGraph(order=3), 3),
])
def test_graph_order(graph, expected):
    order = graph.order
    assert order == expected


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.graph.Graph(), 0),
    (tahini.core.graph.UndirectedGraph(), 0),
    # non empty nodes
    (tahini.core.graph.Graph(nodes=[0, 1]), 0),
    (tahini.core.graph.UndirectedGraph(nodes=[0, 1]), 0),
    # order
    (tahini.core.graph.Graph(order=3), 0),
    (tahini.core.graph.UndirectedGraph(order=3), 0),
    # non empty edges
    (tahini.core.graph.Graph(edges=[(0, 1)]), 1),
    (tahini.core.graph.UndirectedGraph(edges=[(0, 1)]), 1),
    # non empty edges
    (tahini.core.graph.Graph(edges=[(0, 1), (1, 2)]), 2),
    (tahini.core.graph.UndirectedGraph(edges=[(0, 1), (1, 2)]), 2),
])
def test_graph_size(graph, expected):
    size = graph.size
    assert size == expected


@pytest.mark.parametrize('graph, expected', [
    # empty
    (tahini.core.graph.Graph(), f'Graph(nodes={tahini.core.nodes.Nodes()}, edges={tahini.core.edges.Edges()})'),
    (
        tahini.core.graph.UndirectedGraph(),
        f'UndirectedGraph(nodes={tahini.core.nodes.Nodes()}, edges={tahini.core.edges.UndirectedEdges()})',
    ),
])
def test_graph_repr(graph, expected):
    repr_graph = repr(graph)
    assert repr_graph == expected


@pytest.mark.parametrize('graph, args, kwargs, type_error, message_error', [
    # non unique mapping
    (
        tahini.core.graph.Graph(edges=[(0, 1), (0, 2)]),
        [],
        dict(mapper={1: 'a', 2: 'a'}),
        ValueError,
        "Index needs to be unique for 'Nodes'",
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1), (0, 2)]),
        [],
        dict(mapper={1: 'a', 2: 'a'}),
        ValueError,
        "Index needs to be unique for 'Nodes'",
    ),
])
def test_graph_map_nodes_error(graph, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        graph.map_nodes(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (tahini.core.graph.Graph(), [], dict(), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(), tahini.core.graph.UndirectedGraph()),
    # empty mapping
    (tahini.core.graph.Graph(), [], dict(mapper=dict()), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(mapper=dict()), tahini.core.graph.UndirectedGraph()),
    # non empty mapping
    (tahini.core.graph.Graph(), [], dict(mapper={}), tahini.core.graph.Graph()),
    (tahini.core.graph.UndirectedGraph(), [], dict(mapper={}), tahini.core.graph.UndirectedGraph()),
    # non empty graph
    (tahini.core.graph.Graph(order=1), [], dict(mapper={0: 'a'}), tahini.core.graph.Graph(nodes=['a'])),
    (
        tahini.core.graph.UndirectedGraph(order=1),
        [],
        dict(mapper={0: 'a'}),
        tahini.core.graph.UndirectedGraph(nodes=['a']),
    ),
    # non empty graph
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        [],
        dict(mapper={0: 'a', 1: 'b'}),
        tahini.core.graph.Graph(edges=[('a', 'b')]),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
        [],
        dict(mapper={0: 'a', 1: 'b'}),
        tahini.core.graph.UndirectedGraph(edges=[('a', 'b')]),
    ),
])
def test_graph_map_nodes(graph, args, kwargs, expected):
    graph_mapped = graph.map_nodes(*args, **kwargs)
    tahini.testing.testing.assert_graph_equal(graph_mapped, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (
        tahini.core.graph.Graph(),
        [],
        dict(column_name='x'),
        pd.DataFrame(columns=['x'], index=pd.Index([], name=name_nodes)),
    ),
    # no edges
    (
        tahini.core.graph.Graph(nodes=[0]),
        [],
        dict(column_name='x'),
        pd.DataFrame(data=dict(x=[0]), index=pd.Index([0], name=name_nodes)),
    ),
    (
        tahini.core.graph.Graph(nodes=[0, 1]),
        [],
        dict(column_name='x'),
        pd.DataFrame(data=dict(x=[0, 0]), index=pd.Index([0, 1], name=name_nodes)),
    ),
    # edges
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        [],
        dict(column_name='x'),
        pd.DataFrame(data=dict(x=[1, 1]), index=pd.Index([0, 1], name=name_nodes)),
    ),
    (
        tahini.core.graph.Graph(edges=[(0, 1), (0, 2)]),
        [],
        dict(column_name='x'),
        pd.DataFrame(data=dict(x=[2, 1, 1]), index=pd.Index([0, 1, 2], name=name_nodes)),
    ),
    # in
    (
        tahini.core.graph.Graph(edges=[(0, 1), (0, 2)]),
        [],
        dict(column_name='x', columns_stack=[names_edges[1]]),
        pd.DataFrame(data=dict(x=[0, 1, 1]), index=pd.Index([0, 1, 2], name=name_nodes)),
    ),
    # out
    (
        tahini.core.graph.Graph(edges=[(0, 1), (0, 2)]),
        [],
        dict(column_name='x', columns_stack=[names_edges[0]]),
        pd.DataFrame(data=dict(x=[2, 0, 0]), index=pd.Index([0, 1, 2], name=name_nodes)),
    ),
])
def test_graph__edges_stack_value_counts(graph, args, kwargs, expected):
    df = graph._edges_stack_value_counts(*args, **kwargs)
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('graph', [
    tahini.core.graph.Graph(),
    tahini.core.graph.UndirectedGraph(),
])
def test_graph__info_columns_degree(graph):
    info = graph._info_columns_degree()
    assert isinstance(info, dict)
    assert all([item in info for item in ['degree', 'degree_in', 'degree_out']])


@pytest.mark.parametrize('graph, expected', [
    # empty
    (
        tahini.core.graph.Graph(),
        pd.DataFrame(columns=['degree_in', 'degree_out', 'degree'], index=pd.Index([], name=name_nodes)),
    ),
    (tahini.core.graph.UndirectedGraph(), pd.DataFrame(columns=['degree'], index=pd.Index([], name=name_nodes))),
    # non empty
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        pd.DataFrame(
            data=dict(degree_in=[0, 1], degree_out=[1, 0], degree=[1, 1]),
            index=pd.Index([0, 1], name=name_nodes),
        ),
    ),
    # non empty with zero degree
    (
        tahini.core.graph.Graph(nodes=[2], edges=[(0, 1)]),
        pd.DataFrame(
            data=dict(degree_in=[0, 1, 0], degree_out=[1, 0, 0], degree=[1, 1, 0]),
            index=pd.Index([0, 1, 2], name=name_nodes),
        ),
    ),
])
def test_graph_degrees(graph, expected):
    df = graph.degrees
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('graph, expected', [
    # empty
    (
        tahini.core.graph.Graph(),
        pd.DataFrame(columns=['neighbors_in', 'neighbors_out', 'neighbors'], index=pd.Index([], name=name_nodes)),
    ),
    (tahini.core.graph.UndirectedGraph(), pd.DataFrame(columns=['neighbors'], index=pd.Index([], name=name_nodes))),
    # non empty
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        pd.DataFrame(
            data=dict(neighbors_in=[[], [0]], neighbors_out=[[1], []], neighbors=[[1], [0]]),
            index=pd.Index([0, 1], name=name_nodes),
        ),
    ),
    (
        tahini.core.graph.Graph(nodes=[2], edges=[(0, 1)]),
        pd.DataFrame(
            data=dict(neighbors_in=[[], [0], []], neighbors_out=[[1], [], []], neighbors=[[1], [0], []]),
            index=pd.Index([0, 1, 2], name=name_nodes),
        ),
    ),
    (
        tahini.core.graph.Graph(edges=[(0, 1), (1, 0)]),
        pd.DataFrame(
            data=dict(neighbors_in=[[1], [0]], neighbors_out=[[1], [0]], neighbors=[[1], [0]]),
            index=pd.Index([0, 1], name=name_nodes),
        ),
    ),
    (
        tahini.core.graph.Graph(edges=[(0, 1), (1, 2), (0, 2)]),
        pd.DataFrame(
            data=dict(
                neighbors_in=[[], [0], [1, 0]],
                neighbors_out=[[1, 2], [2], []],
                neighbors=[[1, 2], [0, 2], [0, 1]],
            ),
            index=pd.Index([0, 1, 2], name=name_nodes),
        ),
    ),
])
def test_graph_get_neighbors(graph, expected):
    df = graph.neighbors
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('graph, args, kwargs, expected', [
    # empty
    (
        tahini.core.graph.Graph(),
        [],
        dict(),
        pd.DataFrame(index=pd.Index([], name='node_out'), columns=pd.Index([], name='node_in')),
    ),
    (
        tahini.core.graph.UndirectedGraph(),
        [],
        dict(),
        pd.DataFrame(index=pd.Index([], name='node_0'), columns=pd.Index([], name='node_1')),
    ),
    # single node no edges
    (
        tahini.core.graph.Graph(nodes=[0]),
        [],
        dict(),
        pd.DataFrame(data=[[0]], index=pd.Index([0], name='node_out'), columns=pd.Index([0], name='node_in')),
    ),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0]),
        [],
        dict(),
        pd.DataFrame(data=[[0]], index=pd.Index([0], name='node_0'), columns=pd.Index([0], name='node_1')),
    ),
    # two node no edges
    (
        tahini.core.graph.Graph(nodes=[0, 1]),
        [],
        dict(),
        pd.DataFrame(
            data=[[0, 0], [0, 0]],
            index=pd.Index([0, 1], name='node_out'),
            columns=pd.Index([0, 1], name='node_in'),
        ),
    ),
    (
        tahini.core.graph.UndirectedGraph(nodes=[0, 1]),
        [],
        dict(),
        pd.DataFrame(
            data=[[0, 0], [0, 0]],
            index=pd.Index([0, 1], name='node_0'),
            columns=pd.Index([0, 1], name='node_1'),
        ),
    ),
    # single edge
    (
        tahini.core.graph.Graph(edges=[(0, 1)]),
        [],
        dict(),
        pd.DataFrame(
            data=[[0, 1], [0, 0]],
            index=pd.Index([0, 1], name='node_out'),
            columns=pd.Index([0, 1], name='node_in'),
        ),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges=[(0, 1)]),
        [],
        dict(),
        pd.DataFrame(
            data=[[0, 1], [1, 0]],
            index=pd.Index([0, 1], name='node_0'),
            columns=pd.Index([0, 1], name='node_1'),
        ),
    ),
    # weight
    (
        tahini.core.graph.Graph(edges_data=pd.DataFrame(data=dict(weight=[0.1]), index=[(0, 1)])),
        [],
        dict(),
        pd.DataFrame(
            data=[[0, .1], [0, 0]],
            index=pd.Index([0, 1], name='node_out'),
            columns=pd.Index([0, 1], name='node_in'),
        ),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges_data=pd.DataFrame(data=dict(weight=[0.1]), index=[(0, 1)])),
        [],
        dict(),
        pd.DataFrame(
            data=[[0, .1], [.1, 0]],
            index=pd.Index([0, 1], name='node_0'),
            columns=pd.Index([0, 1], name='node_1'),
        ),
    ),
    # weight different column
    (
        tahini.core.graph.Graph(edges_data=pd.DataFrame(data=dict(a=[0.1]), index=[(0, 1)])),
        [],
        dict(weight='a'),
        pd.DataFrame(
            data=[[0, .1], [0, 0]],
            index=pd.Index([0, 1], name='node_out'),
            columns=pd.Index([0, 1], name='node_in'),
        ),
    ),
    (
        tahini.core.graph.UndirectedGraph(edges_data=pd.DataFrame(data=dict(a=[0.1]), index=[(0, 1)])),
        [],
        dict(weight='a'),
        pd.DataFrame(
            data=[[0, .1], [.1, 0]],
            index=pd.Index([0, 1], name='node_0'),
            columns=pd.Index([0, 1], name='node_1'),
        ),
    ),
])
def test_graph_get_adjacency_matrix(graph, args, kwargs, expected):
    df = graph.get_adjacency_matrix(*args, **kwargs)
    assert_frame_equal(df, expected)

import pytest

import tahini.core.edges
import tahini.core.nodes
import tahini.testing


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict()),
    # non empty
    ([], dict(index=[(0, 1)])),
])
def test_edges_init_simple(args, kwargs):
    tahini.core.edges.Edges(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict()),
    # non empty
    ([], dict(index=[(0, 1)])),
])
def test_undirected_edges_init_simple(args, kwargs):
    tahini.core.edges.UndirectedEdges(*args, **kwargs)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.edges.Edges(), [], dict(), tahini.core.nodes.Nodes()),
    (tahini.core.edges.UndirectedEdges(), [], dict(), tahini.core.nodes.Nodes()),
    # non empty
    (tahini.core.edges.Edges(index=[(0, 1)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1])),
    (tahini.core.edges.Edges(index=[(0, 1), (0, 2)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1, 2])),
    (tahini.core.edges.UndirectedEdges(index=[(0, 1)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1])),
    (tahini.core.edges.UndirectedEdges(index=[(0, 1), (0, 2)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1, 2])),
    # order matters
    (tahini.core.edges.Edges(index=[(0, 2), (0, 1)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 2, 1])),
])
def test_edges_nodes(edges, args, kwargs, expected):
    nodes = edges.nodes
    tahini.testing.testing.assert_container_equal(nodes, expected)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.edges.Edges(), [], dict(), tahini.core.edges.Edges()),
    (tahini.core.edges.UndirectedEdges(), [], dict(), tahini.core.edges.UndirectedEdges()),
    # non empty inputs
    (tahini.core.edges.Edges(), [], dict(nodes=[0]), tahini.core.edges.Edges()),
    (tahini.core.edges.UndirectedEdges(), [], dict(nodes=[0]), tahini.core.edges.UndirectedEdges()),
    # non empty
    (tahini.core.edges.Edges(index=[(0, 1)]), [], dict(nodes=[0]), tahini.core.edges.Edges()),
    (tahini.core.edges.UndirectedEdges(index=[(0, 1)]), [], dict(nodes=[0]), tahini.core.edges.UndirectedEdges()),
    (tahini.core.edges.Edges(index=[(0, 1)]), [], dict(nodes=[0, 1]), tahini.core.edges.Edges(index=[(0, 1)])),
    (
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
        [],
        dict(nodes=[0, 1]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
    ),
    (tahini.core.edges.Edges(index=[(0, 1), (0, 2)]), [], dict(nodes=[0, 1]), tahini.core.edges.Edges(index=[(0, 1)])),
    (
        tahini.core.edges.UndirectedEdges(index=[(0, 1), (0, 2)]),
        [],
        dict(nodes=[0, 1]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
    ),
])
def test_edges_keep_nodes(edges, args, kwargs, expected):
    edges_keep_nodes = edges.keep_nodes(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(edges_keep_nodes, expected)

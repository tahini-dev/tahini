import pytest

import tahini.factory
import tahini.core
import tahini.testing


@pytest.mark.parametrize('args, kwargs, expected', [
    # trivial order
    ([], dict(order=0), tahini.core.Graph()),
    ([], dict(order=1), tahini.core.Graph(order=1)),
    # non trivial order
    ([], dict(order=2), tahini.core.Graph(edges=[(0, 1)])),
    ([], dict(order=3), tahini.core.Graph(edges=[(0, 1), (1, 2)])),
    # order and nodes
    ([], dict(order=2, nodes=['a', 'b']), tahini.core.Graph(edges=[('a', 'b')])),
    ([], dict(order=3, nodes=['a', 'b', 'c']), tahini.core.Graph(edges=[('a', 'b'), ('b', 'c')])),
    # trivial nodes
    ([], dict(nodes=[]), tahini.core.Graph()),
    ([], dict(nodes=['a']), tahini.core.Graph(nodes=['a'])),
    # non trivial nodes
    ([], dict(nodes=['a', 'b']), tahini.core.Graph(edges=[('a', 'b')])),
])
def test_get_path(args, kwargs, expected):
    graph = tahini.factory.get_path(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # trivial order
    ([], dict(order=0), tahini.core.Graph()),
    ([], dict(order=1), tahini.core.Graph(order=1)),
    # non trivial order
    ([], dict(order=2), tahini.core.Graph(edges=[(0, 1)])),
    ([], dict(order=3), tahini.core.Graph(edges=[(0, 1), (0, 2)])),
    # order and nodes
    ([], dict(order=2, nodes=['a', 'b']), tahini.core.Graph(edges=[('a', 'b')])),
    ([], dict(order=3, nodes=['a', 'b', 'c']), tahini.core.Graph(edges=[('a', 'b'), ('a', 'c')])),
    # trivial nodes
    ([], dict(nodes=[]), tahini.core.Graph()),
    ([], dict(nodes=['a']), tahini.core.Graph(nodes=['a'])),
    # non trivial nodes
    ([], dict(nodes=['a', 'b']), tahini.core.Graph(edges=[('a', 'b')])),
    ([], dict(nodes=['a', 'b', 'c']), tahini.core.Graph(edges=[('a', 'b'), ('a', 'c')])),
])
def test_get_star(args, kwargs, expected):
    graph = tahini.factory.get_star(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph ,expected)


# todo sort out directed versus undirected graph
@pytest.mark.parametrize('args, kwargs, expected', [
    # trivial order
    ([], dict(order=0), tahini.core.Graph()),
    ([], dict(order=1), tahini.core.Graph(order=1)),
    # non trivial order
    ([], dict(order=2), tahini.core.Graph(edges=[(0, 1), (1, 0)])),
    ([], dict(order=3), tahini.core.Graph(edges=[(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)])),
    # order and nodes
    ([], dict(order=2, nodes=['a', 'b']), tahini.core.Graph(edges=[('a', 'b'), ('b', 'a')])),
    (
        [],
        dict(order=3, nodes=['a', 'b', 'c']),
        tahini.core.Graph(edges=[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]),
    ),
    # trivial nodes
    ([], dict(nodes=[]), tahini.core.Graph()),
    ([], dict(nodes=['a']), tahini.core.Graph(nodes=['a'])),
    # non trivial nodes
    ([], dict(nodes=['a', 'b']), tahini.core.Graph(edges=[('a', 'b'), ('b', 'a')])),
    (
        [],
        dict(nodes=['a', 'b', 'c']),
        tahini.core.Graph(edges=[('a', 'b'), ('a', 'c'), ('b', 'a'), ('b', 'c'), ('c', 'a'), ('c', 'b')]),
    ),
])
def test_get_complete(args, kwargs, expected):
    graph = tahini.factory.get_complete(*args, **kwargs)
    tahini.testing.assert_graph_equal(graph, expected)

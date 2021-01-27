from typing import Optional

from pandas import MultiIndex

from .core import Graph, Nodes

__all__ = [
    'get_path', 'get_star', 'get_complete',
]


def get_path(
        order: Optional[int] = None,
        nodes: Optional[Nodes] = None,
) -> Graph:

    if order is None:
        order = len(nodes)

    graph = Graph(order=order, edges=MultiIndex.from_arrays([range(order - 1), range(1, order)]))

    if nodes is not None:
        graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

    return graph


def get_star(
        order: Optional[int] = None,
        nodes: Optional[Nodes] = None,
) -> Graph:

    if order is None:
        order = len(nodes)

    graph = Graph(order=order, edges=MultiIndex.from_arrays([[0] * (order - 1), range(1, order)]))

    if nodes is not None:
        graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

    return graph


def get_complete(
        order: Optional[int] = None,
        nodes: Optional[Nodes] = None,
) -> Graph:

    if order is None:
        order = len(nodes)

    index = MultiIndex.from_product([range(order)] * 2)
    index = index[index.get_level_values(0) != index.get_level_values(1)]

    graph = Graph(order=order, edges=index)

    if nodes is not None:
        graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

    return graph

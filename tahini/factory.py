from typing import Optional, Union

from pandas import MultiIndex

from .core import Graph, Nodes


def get_path(
        order: int,
        nodes: Optional[Nodes] = None
) -> Graph:

    graph = Graph(order=order, edges=MultiIndex.from_arrays([range(order - 1), range(1, order)]))

    if nodes is not None:
        graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

    return graph


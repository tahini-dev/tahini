from typing import Optional, Union

from pandas import MultiIndex

from .core import Graph, Nodes


def get_path(
        degree: int,
        nodes: Optional[Nodes] = None
) -> Graph:

    graph = Graph(degree=degree, edges=MultiIndex.from_arrays([range(degree - 1), range(1, degree)]))

    if nodes is not None:
        graph = graph.map_nodes(mapper=dict(zip(range(degree), Nodes(index=nodes))))

    return graph

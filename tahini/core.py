from __future__ import annotations
from typing import Optional

from .base import ContainerDataIndexed, TypeIndexInput, TypeDataInput


class Nodes(ContainerDataIndexed):

    def __init__(
            self,
            index: Optional[TypeIndexInput] = None,
            data: Optional[TypeDataInput] = None,
            size: Optional[int] = None,
            **kwargs,
    ):

        if (index is not None and size is not None) or (data is not None and size is not None):
            raise ValueError(
                f"Inputs for '{self.__class__.__name__}' can either be empty or contain "
                f"'index', "
                f"'data', "
                f"'index' and 'data' "
                f"or 'size'"
            )

        if size is not None:
            index = range(size)

        super().__init__(index=index, data=data, **kwargs)

    @staticmethod
    def _name_index():
        return 'node'




# class Edges:
#
#     def __init__(
#             self,
#             edges: Optional[TypeEdgeInput] = None,
#             data: Optional[TypeEdgesDataInput] = None,
#             **kwargs,
#     ):
#
#         if isinstance(data, DataFrame) and edges is not None:
#             raise ValueError(f"'edges' has to be None if 'data' is a DataFrame for '{self.__class__.__name__}'")
#
#         if isinstance(edges, Edges):
#             data = edges.data
#             edges = None
#
#         if isinstance(data, DataFrame) and not data.empty:
#             edges = data.index
#
#         if edges is not None:
#             edges = MultiIndex.from_tuples(edges)
#
#         self.data = DataFrame(data=data, index=edges, **kwargs)
#
#
# TypeEdgeInput = Union[Edges, MultiIndex, Iterable]
# TypeEdgesDataInput = Union[DataFrame, Dict, Iterable]

# class Graph:
#
#     def __init__(
#             self,
#             nodes: Optional[Union[Nodes, Index, Iterable]] = None,
#             nodes_data: Optional[Union[DataFrame, Dict, Iterable]] = None,
#             degree: Optional[int] = None,
#     ):
#
#         self.nodes = Nodes(nodes=nodes, data=nodes_data, size=degree)

from __future__ import annotations
from collections.abc import Collection, Iterable
from typing import Optional, Union, Dict, Any, Callable

from pandas import DataFrame, Index, RangeIndex


class Nodes(Collection):

    def __init__(
            self,
            nodes: Optional[TypeNodesInput] = None,
            data: Optional[TypeNodesDataInput] = None,
            size: Optional[int] = None,
            **kwargs,
    ):

        if (nodes is not None and size is not None) or (data is not None and size is not None):
            raise ValueError(
                f"Inputs for '{self.__class__.__name__}' can either be empty or contain "
                f"'nodes', "
                f"'data', "
                f"'nodes' and 'data' "
                f"or 'size'"
            )

        if isinstance(nodes, Nodes):
            data = nodes.data
            nodes = None

        if size is not None:
            nodes = RangeIndex(stop=size)

        if isinstance(data, DataFrame) and nodes is not None:
            raise ValueError(f"'nodes' has to be None if 'data' is a DataFrame for '{self.__class__.__name__}'")

        self.data = DataFrame(data=data, index=nodes, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}(index={self.data.index})'

    def __iter__(self):
        return iter(self.data.index)

    def __contains__(
            self,
            item: Any,
    ):
        return item in self.data.index

    def __len__(self):
        return len(self.data.index)

    def __eq__(
            self,
            other: Any,
    ):
        return isinstance(other, Nodes) and len(self.data.index.symmetric_difference(other.data.index)) == 0

    def update(
            self,
            nodes: Optional[TypeNodesInput] = None,
            data: Optional[TypeNodesDataInput] = None,
            func: Optional[Callable] = None,
            **kwargs,
    ) -> Nodes:
        other = Nodes(nodes=nodes, data=data)
        if func is None:
            func = _update_func
        self.data = self.data.combine(other.data, func, **kwargs)
        return self

    def drop(
            self,
            nodes: Optional[TypeNodesInput] = None,
            **kwargs,
    ) -> Nodes:
        self.data = self.data.drop(index=nodes, **kwargs)
        return self


TypeNodesInput = Union[Nodes, Index, Iterable]
TypeNodesDataInput = Union[DataFrame, Dict, Iterable]


def _update_func(s1, s2):
    return s2.fillna(s1)


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

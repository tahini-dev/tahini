from __future__ import annotations
from typing import Optional, Union, TypeVar, NoReturn
from collections.abc import Sequence

from .base import ContainerDataIndexed, ContainerDataIndexedMulti, TypeIndexMultiInput, TypeIndexInput, TypeDataInput


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
    def _names_index() -> str:
        return 'node'


class Edges(ContainerDataIndexedMulti):

    @staticmethod
    def _names_index() -> Sequence[str]:
        return ['node_1', 'node_2']

    def get_nodes(self) -> Nodes:
        return Nodes(index=self.data.index.to_frame().stack().drop_duplicates())


# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
T = TypeVar('T', bound='Graph')
TypeNodesInput = Union[Nodes, TypeIndexInput]
TypeEdgesInput = Union[Edges, TypeIndexMultiInput]


class Graph:

    def __init__(
            self,
            nodes: Optional[TypeNodesInput] = None,
            edges: Optional[TypeEdgesInput] = None,
            degree: Optional[int] = None,
            nodes_data: Optional[TypeDataInput] = None,
            edges_data: Optional[TypeDataInput] = None,
            **kwargs,
    ):
        self._nodes = Nodes(index=nodes, data=nodes_data, size=degree, **kwargs)
        self._edges = Edges(index=edges, data=edges_data, **kwargs)
        self._nodes = self._update_nodes_from_edges()

    @property
    def nodes(self) -> Nodes:
        return self._nodes

    @property
    def edges(self) -> Edges:
        return self._edges

    def update_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> T:
        self._nodes = self._nodes.update(index=nodes, data=data, **kwargs)
        return self

    def _update_nodes_from_edges(self) -> Nodes:
        return self._nodes.update(index=self._edges.get_nodes())

    def update_edges(
            self,
            edges: Optional[TypeEdgesInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> T:
        self._edges = self._edges.update(index=edges, data=data, **kwargs)
        self._nodes = self._update_nodes_from_edges()
        return self

    def update(
            self,
            nodes: Optional[TypeNodesInput] = None,
            edges: Optional[TypeEdgesInput] = None,
            nodes_data: Optional[TypeDataInput] = None,
            edges_data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> T:
        graph = self.update_nodes(nodes=nodes, data=nodes_data, **kwargs)
        graph = graph.update_edges(edges=edges, data=edges_data, **kwargs)
        return graph


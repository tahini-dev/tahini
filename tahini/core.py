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


# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
TypeEdges = TypeVar('TypeEdges', bound='Edges')


class Edges(ContainerDataIndexedMulti):

    @staticmethod
    def _names_index() -> Sequence[str]:
        return ['node_1', 'node_2']

    def get_nodes(self) -> Nodes:
        return Nodes(index=self.data.index.to_frame().stack().drop_duplicates())

    def keep_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
    ) -> TypeEdges:
        if nodes is not None:
            nodes = Nodes(index=nodes)
            self.data = self.data[
                lambda x: x.index.get_level_values(0).isin(nodes) & x.index.get_level_values(1).isin(nodes)
            ]
        return self


# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
TypeGraph = TypeVar('TypeGraph', bound='Graph')
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

    @nodes.setter
    def nodes(
            self,
            value: TypeNodesInput,
    ):
        self._nodes = Nodes(index=value)
        self._update_edges_from_nodes()

    @property
    def edges(self) -> Edges:
        return self._edges

    @edges.setter
    def edges(
            self,
            value: TypeEdgesInput,
    ):
        self._edges = Edges(index=value)
        self._update_nodes_from_edges()

    def _update_nodes_from_edges(self) -> Nodes:
        return self.nodes.update(index=self.edges.get_nodes())

    def _update_edges_from_nodes(self) -> Nodes:
        return self.edges.keep_nodes(nodes=self.nodes)

    def update_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> TypeGraph:
        self.nodes = self.nodes.update(index=nodes, data=data, **kwargs)
        return self

    def update_edges(
            self,
            edges: Optional[TypeEdgesInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> TypeGraph:
        self.edges = self.edges.update(index=edges, data=data, **kwargs)
        return self

    def update(
            self,
            nodes: Optional[TypeNodesInput] = None,
            edges: Optional[TypeEdgesInput] = None,
            nodes_data: Optional[TypeDataInput] = None,
            edges_data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> TypeGraph:
        graph = self.update_nodes(nodes=nodes, data=nodes_data, **kwargs)
        graph = graph.update_edges(edges=edges, data=edges_data, **kwargs)
        return graph

    def drop_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
            **kwargs,
    ) -> TypeGraph:
        self.nodes = self.nodes.drop(index=nodes, **kwargs)
        return self

    def drop_edges(
            self,
            edges: Optional[TypeEdgesInput] = None,
            **kwargs,
    ) -> TypeGraph:
        self.edges = self.edges.drop(index=edges, **kwargs)
        return self

    def drop(
            self,
            nodes: Optional[TypeNodesInput] = None,
            edges: Optional[TypeEdgesInput] = None,
            **kwargs,
    ) -> TypeGraph:
        graph = self.drop_nodes(nodes=nodes, **kwargs)
        graph = graph.drop_edges(edges=edges, **kwargs)
        return graph

    @property
    def degree(self) -> int:
        return len(self.nodes)

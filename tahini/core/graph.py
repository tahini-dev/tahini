from __future__ import annotations
from typing import Optional, Union, TypeVar, Callable

from pandas import Series, MultiIndex

from .base import (
    TypeDataInput,
    TypeMapper,
)
from .nodes import (
    Nodes,
    TypeNodesInput
)
from .edges import (
    Edges,
    UndirectedEdges,
    TypeEdgesInput,
)

__all__ = [
    'Graph',
    'UndirectedGraph',
    'TypeDataInput',
]

TypeGraph = TypeVar('TypeGraph', bound='Graph')


class Graph:

    _type_edges = Edges

    def __init__(
            self,
            nodes: Optional[TypeNodesInput] = None,
            edges: Optional[TypeEdgesInput] = None,
            order: Optional[int] = None,
            nodes_data: Optional[TypeDataInput] = None,
            edges_data: Optional[TypeDataInput] = None,
            **kwargs,
    ):
        self._nodes = Nodes(index=nodes, data=nodes_data, order=order, **kwargs)
        self._edges = self._type_edges(index=edges, data=edges_data, **kwargs)
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
        self._edges = self._type_edges(index=value)
        self._update_nodes_from_edges()

    def assign(
            self,
            nodes: Optional[Union[Nodes, Callable]] = None,
            edges: Optional[Union[Edges, Callable]] = None,
    ) -> TypeGraph:
        if nodes is not None:
            if isinstance(nodes, Callable):
                nodes = nodes(self)
            self.nodes = nodes

        if edges is not None:
            if isinstance(edges, Callable):
                edges = edges(self)
            self.edges = edges

        return self

    def _update_nodes_from_edges(self) -> Nodes:
        return self.nodes.update(index=self.edges.nodes)

    def _update_edges_from_nodes(self) -> Edges:
        return self.edges.keep_nodes(nodes=self.nodes)

    def update_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> TypeGraph:
        return self.assign(nodes=self.nodes.update(index=nodes, data=data, **kwargs))

    def update_edges(
            self,
            edges: Optional[TypeEdgesInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ) -> TypeGraph:
        return self.assign(edges=self.edges.update(index=edges, data=data, **kwargs))

    def drop_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
            **kwargs,
    ) -> TypeGraph:
        self.assign(nodes=self.nodes.drop(index=nodes, **kwargs))
        return self

    def drop_edges(
            self,
            edges: Optional[TypeEdgesInput] = None,
            **kwargs,
    ) -> TypeGraph:
        return self.assign(edges=self.edges.drop(index=edges, **kwargs))

    @property
    def order(self) -> int:
        return len(self.nodes)

    @property
    def size(self) -> int:
        return len(self.edges)

    def __repr__(self):
        return f'{self.__class__.__name__}(nodes={self.nodes}, edges={self.edges})'

    def map_nodes(
            self,
            mapper: Optional[TypeMapper] = None,
            **kwargs
    ) -> TypeGraph:
        self._nodes = self.nodes.map(mapper=mapper, **kwargs)
        self._edges = self.edges.map(mapper=mapper, **kwargs)
        return self

    def get_degrees(self) -> TypeGraph:

        degree_by_node = (
            self.edges.data_internal
            [self.edges.names_index]
            .stack()
            .value_counts()
            .rename_axis(index=self.nodes.names_index[0])
            .rename('degree')
        )

        graph = self.update_nodes(
            data=(
                self.nodes
                .update(data=degree_by_node)
                .data
                .assign(
                    degree=lambda x: x['degree'].fillna(0),
                )
            )
        )

        return graph

    # def get_neighbors(
    #         self,
    # ) -> Series:
    #
    #     neighbors = (
    #         self.edges
    #         .data
    #         .reset_index(level=self.edges.names_index[0])
    #         .groupby(level=self.edges.names_index[1])
    #         [self.edges.names_index[0]]
    #         .apply(list)
    #         .rename_axis(index='node')
    #         .rename('neighbors')
    #     )
    #
    #     neighbors = (
    #         neighbors
    #         .combine_first(Series(index=self.nodes.data.index, name='neighbors'))
    #         [self.nodes]
    #     )
    #
    #     return neighbors

    @classmethod
    def path(
            cls,
            order: Optional[int] = None,
            nodes: Optional[Nodes] = None,
    ) -> TypeGraph:

        if order is None:
            order = len(nodes)

        graph = cls(order=order, edges=MultiIndex.from_arrays([range(order - 1), range(1, order)]))

        if nodes is not None:
            graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

        return graph

    @classmethod
    def cycle(
            cls,
            order: Optional[int] = None,
            nodes: Optional[Nodes] = None,
    ) -> TypeGraph:

        if order is None:
            order = len(nodes)

        if order < 3:
            raise ValueError("Inputs 'order' or length of 'nodes' has to be >= 3 for cycle")

        nodes_left = range(order)
        nodes_right = list(range(1, order))
        nodes_right.append(0)
        edges = MultiIndex.from_arrays([nodes_left, nodes_right])

        graph = cls(order=order, edges=edges)

        if nodes is not None:
            graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

        return graph

    @classmethod
    def star(
            cls,
            order: Optional[int] = None,
            nodes: Optional[Nodes] = None,
    ) -> TypeGraph:

        if order is None:
            order = len(nodes)

        graph = cls(order=order, edges=MultiIndex.from_arrays([[0] * (order - 1), range(1, order)]))

        if nodes is not None:
            graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

        return graph

    @classmethod
    def _get_unique_edges(
            cls,
            edges: MultiIndex,
    ) -> MultiIndex:
        return edges

    @classmethod
    def complete(
            cls,
            order: Optional[int] = None,
            nodes: Optional[Nodes] = None,
    ) -> TypeGraph:

        if order is None:
            order = len(nodes)

        edges = MultiIndex.from_product([range(order)] * 2)
        edges = edges[edges.get_level_values(0) != edges.get_level_values(1)]
        edges = cls._get_unique_edges(edges=edges)

        graph = cls(order=order, edges=edges)

        if nodes is not None:
            graph = graph.map_nodes(mapper=dict(zip(range(order), Nodes(index=nodes))))

        return graph


class UndirectedGraph(Graph):
    _type_edges = UndirectedEdges

    @classmethod
    def _get_unique_edges(
            cls,
            edges: MultiIndex,
    ) -> MultiIndex:
        return edges.map(frozenset).drop_duplicates().map(tuple)

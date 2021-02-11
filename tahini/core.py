from __future__ import annotations
from typing import Optional, Union, TypeVar

from pandas import Series

from .container import (
    ContainerDataIndexed,
    ContainerDataIndexedMulti,
    ContainerDataIndexedMultiSets,
    TypeIndexInput,
    TypeIndexMultiInput,
    TypeDataInput,
    TypeMapper,
)

__all__ = [
    'Graph',
    'UndirectedGraph',
    'Nodes',
    'Edges',
    'UndirectedEdges',
    'TypeNodesInput',
    'TypeEdgesInput',
    'TypeDataInput',
]


class Nodes(ContainerDataIndexed):

    _names_index = ['node']
    _name_index_internal = 'node_internal'

    def __init__(
            self,
            index: Optional[TypeIndexInput] = None,
            data: Optional[TypeDataInput] = None,
            order: Optional[int] = None,
            **kwargs,
    ):

        if order is not None:
            index = range(order)

        super().__init__(index=index, data=data, **kwargs)


TypeEdges = TypeVar('TypeEdges', bound='Edges')


class Edges(ContainerDataIndexedMulti):

    _names_index = ['node_from', 'node_to']
    _name_index_internal = 'edge_internal'

    def get_nodes(self) -> Nodes:
        return Nodes(index=self.data_internal[self._names_index].stack().drop_duplicates())

    def keep_nodes(
            self,
            nodes: Optional[TypeNodesInput] = None,
    ) -> TypeEdges:
        if nodes is not None:
            nodes = Nodes(index=nodes)
            self.data = self.data_internal[
                lambda x: x[self._names_index[0]].isin(nodes) & x[self._names_index[1]].isin(nodes)
            ]
        return self


class UndirectedEdges(ContainerDataIndexedMultiSets, Edges):
    _names_index = ['node_0', 'node_1']
    _name_index_internal = 'edge_internal'


TypeGraph = TypeVar('TypeGraph', bound='Graph')
TypeNodesInput = Union[Nodes, TypeIndexInput]
TypeEdgesInput = Union[Edges, TypeIndexMultiInput]


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

    def _update_nodes_from_edges(self) -> Nodes:
        return self.nodes.update(index=self.edges.get_nodes())

    def _update_edges_from_nodes(self) -> Edges:
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

    # need to keep all three update functions for kwargs
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
        self._nodes = self._nodes.map(mapper=mapper, **kwargs)
        self._edges = self._edges.map(mapper=mapper, **kwargs)
        return self

    def get_degree_by_node(self) -> Series:

        degree_by_node = (
            self.edges.data.index
            .to_frame()
            .stack()
            .value_counts()
            .rename_axis(index='node')
            .rename('degree')
        )

        degree_by_node = (
            degree_by_node
            .combine_first(Series(data=0, index=self.nodes.data.index, name='degree'))
            [self.nodes]
            .astype('int64')
        )

        return degree_by_node

    def get_neighbors(
            self,
    ) -> Series:

        neighbors = (
            self.edges
            .data
            .reset_index(level='node_to')
            .groupby(level='node_from')
            ['node_to']
            .apply(list)
            .rename_axis(index='node')
            .rename('neighbors')
        )

        neighbors = (
            neighbors
            .combine_first(Series(index=self.nodes.data.index, name='neighbors'))
            [self.nodes]
        )

        return neighbors


class UndirectedGraph(Graph):
    _type_edges = UndirectedEdges

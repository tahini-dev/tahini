from __future__ import annotations
from typing import Optional, Union, TypeVar

from .base import (
    ContainerDataIndexedMulti,
    ContainerDataIndexedMultiSets,
    TypeIndexMultiInput,
)
from .nodes import (
    Nodes,
    TypeNodesInput
)

__all__ = [
    'Edges',
    'UndirectedEdges',
    'TypeEdgesInput',
]

TypeEdges = TypeVar('TypeEdges', bound='Edges')


class Edges(ContainerDataIndexedMulti):

    _names_index = ['node_in', 'node_out']
    _name_index_internal = 'edge_internal'

    @property
    def nodes(self) -> Nodes:
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


TypeEdgesInput = Union[Edges, TypeIndexMultiInput]

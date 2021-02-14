from __future__ import annotations
from typing import Optional, Union

from .base import (
    ContainerDataIndexed,
    TypeIndexInput,
    TypeDataInput,
)

__all__ = [
    'Nodes',
    'TypeNodesInput',
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


TypeNodesInput = Union[Nodes, TypeIndexInput]

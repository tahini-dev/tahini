from __future__ import annotations
from typing import Optional, Union, Sequence

from pandas import DataFrame

from .base import (
    ContainerDataIndexed,
    TypeIndexInput,
    TypeDataInput,
)
from ..plot.positions import get as get_positions

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

    def get_positions(
            self,
            layout: Optional[str] = None,
            center: Optional[Sequence] = None,
            dim: Optional[int] = None,
            **kwargs,
    ) -> DataFrame:

        df = self.__class__(data=get_positions(
            items=self.data.index,
            layout=layout,
            center=center,
            dim=dim,
            **kwargs,
        )).data

        return df


TypeNodesInput = Union[Nodes, TypeIndexInput]

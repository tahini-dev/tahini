from __future__ import annotations
from typing import Optional, Union, Sequence

from pandas import DataFrame, MultiIndex

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

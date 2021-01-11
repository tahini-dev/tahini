from __future__ import annotations
from collections.abc import Collection, Iterable
from typing import Optional, Union, Dict, Any, Callable, TypeVar

from pandas import DataFrame, Series, Index, MultiIndex

# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
T = TypeVar('T', bound='ContainerDataIndexed')
TypeIndexInput = Union[T, Index, Iterable]
TypeDataInput = Union[DataFrame, Dict, Iterable]


class ContainerDataIndexed(Collection):

    def __init__(
            self,
            index: Optional[TypeIndexInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ):

        if index is not None and isinstance(data, DataFrame):
            raise ValueError(
                f"If input 'data' is 'pandas.DataFrame' then input 'index' has to be 'None' "
                f"for initializing '{self.__class__.__name__}'"
            )

        if isinstance(index, self.__class__):
            data = index.data
            index = None

        if index is not None:
            index = self._create_index(index)

        self.data = DataFrame(data=data, index=index, **kwargs)

        self._validate_data()

    @staticmethod
    def _create_index(index):
        return Index(index)

    def _validate_data(self):
        self.data = self.data.rename_axis(index=self._name_index())

    def __repr__(self):
        return f'{self.__class__.__name__}(index={self.data.index})'

    @staticmethod
    def _name_index():
        return None

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
        return isinstance(other, self.__class__) and len(self.data.index.symmetric_difference(other.data.index)) == 0

    def update(
            self,
            index: Optional[TypeIndexInput] = None,
            data: Optional[TypeDataInput] = None,
            func: Optional[Callable] = None,
            **kwargs,
    ) -> T:
        other = self.__class__(index=index, data=data)
        if func is None:
            func = self._update_func
        self.data = self.data.combine(other.data, func, **kwargs)
        return self

    @staticmethod
    def _update_func(
            s1: Series,
            s2: Series,
    ) -> Series:
        return s2.fillna(s1)

    def drop(
            self,
            index: Optional[TypeIndexInput] = None,
            **kwargs,
    ) -> T:
        self.data = self.data.drop(index=index, **kwargs)
        return self

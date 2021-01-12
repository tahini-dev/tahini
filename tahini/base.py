from __future__ import annotations
from collections.abc import Collection, Iterable, Sequence
from typing import Optional, Union, Dict, Any, Callable, TypeVar, Iterable as TypeIterable

from pandas import DataFrame, Series, Index, MultiIndex

# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
T = TypeVar('T', bound='ContainerDataIndexed')
TypeIndexInput = Union[T, Index, Iterable]
TypeIndexMultiInput = Union[T, MultiIndex, TypeIterable[Sequence]]
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

    def _create_index(
            self,
            index: TypeIndexInput,
    ) -> Index:
        return Index(index)

    @staticmethod
    def _names_index() -> Optional[Union[str, Sequence[str]]]:
        return None

    def _validate_data(self):
        self.data = self.data.rename_axis(index=self._names_index())

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
        if index is None:
            index = []
        self.data = self.data.drop(index=index, **kwargs)
        return self


class ContainerDataIndexedMulti(ContainerDataIndexed):

    def __init__(
            self,
            index: Optional[TypeIndexMultiInput] = None,
            **kwargs,
    ):
        super().__init__(index=index, **kwargs)

    @staticmethod
    def _names_index():
        return [None, None]

    def _create_index(
            self,
            index: TypeIndexMultiInput,
    ) -> MultiIndex:
        names_index = self._names_index()
        num_levels = len(names_index)
        if len(index) == 0:
            index = MultiIndex(levels=[[]] * num_levels, codes=[[]] * num_levels)
        else:
            index = MultiIndex.from_tuples(index)
        index = index.rename(names_index)
        return index

    def _validate_data(self):
        if not isinstance(self.data.index, MultiIndex):
            self.data = self.data.set_index(self._create_index(self.data.index))
        self.data = self.data.rename_axis(index=self._names_index())

from __future__ import annotations
from collections.abc import Collection, Iterable, Sequence
from typing import (
    Optional, Union, Dict, Any, Callable, TypeVar, Iterable as TypeIterable, Sequence as TypeSequence, Hashable
)

from pandas import DataFrame, Series, Index, MultiIndex

__all__ = [
    'ContainerDataIndexed', 'ContainerDataIndexedMulti', 'TypeIndexMultiInput', 'TypeIndexInput', 'TypeDataInput',
    'TypeMapper',
]

# https://www.python.org/dev/peps/pep-0484/#annotating-instance-and-class-methods
TypeContainerDataIndexed = TypeVar('TypeContainerDataIndexed', bound='ContainerDataIndexed')
TypeIndexInput = Union[TypeContainerDataIndexed, Index, TypeIterable[Hashable]]
TypeIndexMultiInput = Union[TypeContainerDataIndexed, MultiIndex, TypeIterable[TypeSequence[Hashable]]]
TypeDataInput = Union[DataFrame, Dict, Iterable]
TypeMapper = Union[Callable, Dict, Series]


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
        if not self.data.index.is_unique:
            raise ValueError(f"Index needs to be unique for '{self.__class__.__name__}'")

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
    ) -> TypeContainerDataIndexed:
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
    ) -> TypeContainerDataIndexed:
        if index is None:
            index = []
        self.data = self.data.drop(index=index, **kwargs)
        return self

    def map(
            self,
            mapper: Optional[TypeMapper] = None,
            **kwargs,
    ) -> TypeContainerDataIndexed:
        if mapper is not None:
            self._map(mapper=mapper, **kwargs)
        self._validate_data()
        return self

    def _map(
            self,
            mapper: TypeMapper,
            **kwargs,
    ):
        self.data.index = self.data.index.map(mapper=mapper, **kwargs)

    def copy(
            self,
            *args,
            **kwargs,
    ) -> TypeContainerDataIndexed:
        return self.__class__(data=self.data.copy(*args, **kwargs))


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

        if not self.data.index.to_flat_index().is_unique:
            raise ValueError(f"Index needs to be unique for '{self.__class__.__name__}'")

    def _map(
            self,
            mapper: TypeMapper,
            **kwargs,
    ) -> TypeContainerDataIndexed:

        index = [
            self.data.index.get_level_values(level=level).map(mapper, **kwargs)
            for level in range(self.data.index.nlevels)
        ]

        self.data.index = self._create_index(index=MultiIndex.from_arrays(index))

        return self

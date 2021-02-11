from __future__ import annotations
from collections.abc import Collection, Iterable
from typing import (
    Optional, Union, Dict, Any, Callable, TypeVar, Iterable as TypeIterable, Sequence as TypeSequence, Hashable,
    NoReturn,
)

from pandas import DataFrame, Series, Index, MultiIndex

__all__ = [
    'ContainerDataIndexed',
    'ContainerDataIndexedMulti',
    'ContainerDataIndexedMultiSets',
    'TypeIndexInput',
    'TypeIndexMultiInput',
    'TypeDataInput',
    'TypeMapper',
]

TypeContainerDataIndexed = TypeVar('TypeContainerDataIndexed', bound='ContainerDataIndexed')
TypeIndexInput = Union[TypeContainerDataIndexed, Index, TypeIterable[Hashable]]
TypeIndexMultiInput = Union[TypeContainerDataIndexed, MultiIndex, TypeIterable[TypeSequence[Hashable]]]
TypeDataInput = Union[DataFrame, Dict, Iterable]
TypeMapper = Union[Callable, Dict, Series]


class ContainerDataIndexed(Collection):

    _names_index = ['index']
    _name_index_internal = 'index_internal'

    def __init__(
            self,
            index: Optional[TypeIndexInput] = None,
            data: Optional[TypeDataInput] = None,
            **kwargs,
    ):

        if isinstance(index, self.__class__):
            data = index.data
            index = None

        if index is not None and not isinstance(index, Index):
            index = self._validate_index(index=index)

        if data is not None:
            data = DataFrame(data=data, **kwargs)
            if index is not None:
                data.index = index
        else:
            data = DataFrame(index=index, **kwargs)

        self.data = data

    @property
    def data_internal(
            self,
    ) -> DataFrame:
        return self._data

    @property
    def data(
            self,
    ) -> DataFrame:
        return self.data_internal.set_index(keys=self._names_index)

    @data.setter
    def data(
            self,
            value: DataFrame,
    ) -> NoReturn:

        if len(value.columns.intersection(self._names_index)) == len(self._names_index):
            value = value.set_index(self._names_index)

        data = (
            self._validate_data(data=value)
            .assign(**{self._name_index_internal: lambda x: self._create_index_internal(x.index)})
            .reset_index()
            .set_index(self._name_index_internal)
        )

        self._data = data

    @property
    def data_testing(
            self,
    ) -> DataFrame:
        return self.data_internal.drop(columns=self._names_index)

    @classmethod
    def _validate_data(
            cls,
            data: DataFrame,
    ) -> DataFrame:
        data.index = cls._validate_index(index=data.index)
        return data

    @classmethod
    def _validate_index(
            cls,
            index: Union[TypeIndexInput, TypeIndexMultiInput],
    ) -> Union[Index, MultiIndex]:
        if not isinstance(index, Index):
            index = Index(index)
        index_internal = cls._create_index_internal(index=index)
        if not index_internal.is_unique:
            raise ValueError(f"Index needs to be unique for '{cls.__name__}'")
        index.names = cls._names_index
        return index

    @classmethod
    def _create_index_internal(
            cls,
            index: Union[Index, MultiIndex],
    ) -> Index:
        return index.rename(cls._name_index_internal)

    def drop(
            self,
            index: Optional[Union[TypeIndexInput, TypeIndexMultiInput]] = None,
            **kwargs,
    ) -> TypeContainerDataIndexed:
        if index is None:
            index = []
        self.data = self.data_internal.drop(index=self.__class__(index=index).data_internal.index, **kwargs)
        return self

    def update(
            self,
            index: Optional[Union[TypeIndexInput, TypeIndexMultiInput]] = None,
            data: Optional[TypeDataInput] = None,
            func: Optional[Callable] = None,
            **kwargs,
    ) -> TypeContainerDataIndexed:

        if func is None:
            func = self._update_func

        self.data = (
            self.data_internal
            .combine(
                other=self.__class__(index=index, data=data).data_internal,
                func=func,
                **kwargs,
            )
        )

        return self

    @staticmethod
    def _update_func(
            s1: Series,
            s2: Series,
    ) -> Series:
        return s2.fillna(s1)

    def map(
            self,
            mapper: Optional[TypeMapper] = None,
            **kwargs,
    ) -> TypeContainerDataIndexed:
        if mapper is not None:
            self.data = (
                self.data_internal
                .assign(**{
                    column: self.data_internal[column].map(mapper, **kwargs)
                    for column in self._names_index
                })
            )
        return self

    def __repr__(self):
        return f'{self.__class__.__name__}(index={self.data.index})'

    def iter(self) -> Iterable:
        return iter(self.data_internal.index)

    def __iter__(self):
        return self.iter()

    def __contains__(
            self,
            item: Any,
    ):
        return self.__class__(index=[item]).data_internal.index[0] in self.data_internal.index

    def __len__(self):
        return len(self.data_internal.index)

    def __eq__(
            self,
            other: Any,
    ):
        return (
            isinstance(other, self.__class__) and
            len(self.data_internal.index.symmetric_difference(other.data_internal.index)) == 0
        )


class ContainerDataIndexedMulti(ContainerDataIndexed):

    _names_index = ['index_0', 'index_1']

    def __init__(
            self,
            index: Optional[TypeIndexMultiInput] = None,
            **kwargs,
    ):
        super().__init__(index=index, **kwargs)

    @classmethod
    def _create_index_internal(
            cls,
            index: MultiIndex,
    ) -> Index:
        return index.to_flat_index().rename(cls._name_index_internal)

    @classmethod
    def _validate_index(
            cls,
            index: Union[TypeIndexInput, TypeIndexMultiInput]
    ) -> MultiIndex:

        num_levels = len(cls._names_index)

        if len(index) == 0:
            index = MultiIndex(levels=[[]] * num_levels, codes=[[]] * num_levels)
        else:
            index = MultiIndex.from_tuples(index)

        index = super()._validate_index(index=index)

        return index


class ContainerDataIndexedMultiSets(ContainerDataIndexedMulti):

    @classmethod
    def _create_index_internal(
            cls,
            index: MultiIndex,
    ) -> Index:
        return index.to_flat_index().map(frozenset).rename(cls._name_index_internal)

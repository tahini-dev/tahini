from typing import Optional, Union, Iterable, Dict

from pandas import DataFrame, Index, RangeIndex


class Nodes:

    def __init__(
            self,
            data: Optional[Union[DataFrame, Dict, Iterable]] = None,
            index: Optional[Union[Index, Iterable]] = None,
            size: Optional[int] = None,
            **kwargs,
    ):

        if data is None and index is None and size is None:
            index = []

        if size is not None:
            index = RangeIndex(stop=size)

        self.data = DataFrame(data=data, index=index, **kwargs)

    def __repr__(self):
        return f'{self.__class__.__name__}(index={self.data.index})'

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
            raise ValueError(
                f'Need to provide at least "data" or "index" or "size" for initializing "{self.__class__.__name__}".'
            )

        if size is not None:
            index = RangeIndex(stop=size)

        self.data = DataFrame(data=data, index=index, **kwargs)

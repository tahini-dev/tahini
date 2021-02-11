from typing import Optional, Union, NoReturn

import pandas as pd

from .container import ContainerDataIndexed
from .core import Graph

__all__ = []


def assert_container_equal(
        left: ContainerDataIndexed,
        right: ContainerDataIndexed,
        check_dtype: Optional[bool] = None,
        check_index_type: Optional[Union[bool, str]] = None,
        check_column_type: Optional[Union[bool, str]] = None,
        check_like: Optional[bool] = None,
        obj: Optional[str] = None,
        **kwargs,
) -> NoReturn:

    if check_dtype is None:
        check_dtype = False
    if check_index_type is None:
        check_index_type = False
    if check_column_type is None:
        check_column_type = False
    if check_like is None:
        check_like = True

    assert type(left) is type(right), f"""Types are different

[left]:  {left.__class__.__name__}
[right]: {right.__class__.__name__}"""

    if obj is None:
        obj = f'{left.__class__.__name__}.data_testing'

    pd.testing.assert_frame_equal(
        left=left.data_testing,
        right=right.data_testing,
        check_dtype=check_dtype,
        check_index_type=check_index_type,
        check_column_type=check_column_type,
        check_like=check_like,
        obj=obj,
        **kwargs,
    )


def assert_graph_equal(
        left: Graph,
        right: Graph,
        obj: Optional[str] = None,
        **kwargs,
) -> NoReturn:

    if obj is not None:
        flag_obj = True
    else:
        flag_obj = False

    if not flag_obj:
        obj = f'{left.__class__.__name__}.nodes.data_testing'
    assert_container_equal(left=left.nodes, right=right.nodes, obj=obj, **kwargs)

    if not flag_obj:
        obj = f'{left.__class__.__name__}.edges.data_testing'
    assert_container_equal(left=left.edges, right=right.edges, obj=obj, **kwargs)

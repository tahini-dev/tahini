from typing import Optional, Union, NoReturn

import pandas as pd

from .core import Nodes, Edges, Graph

__all__ = ['assert_graph_equal', 'assert_nodes_equal', 'assert_edges_equal']


def assert_nodes_equal(
        left: Nodes,
        right: Nodes,
        check_index_type: Optional[Union[bool, str]] = None,
        obj: Optional[str] = None,
        **kwargs,
) -> NoReturn:

    if check_index_type is None:
        check_index_type = False

    if obj is None:
        obj = 'Nodes.data'

    pd.testing.assert_frame_equal(
        left=left.data,
        right=right.data,
        check_index_type=check_index_type,
        obj=obj,
        **kwargs,
    )


def assert_edges_equal(
        left: Edges,
        right: Edges,
        check_index_type: Optional[Union[bool, str]] = None,
        obj: Optional[str] = None,
        **kwargs,
) -> NoReturn:

    if check_index_type is None:
        check_index_type = False

    if obj is None:
        obj = 'Edges.data'

    try:
        pd.testing.assert_frame_equal(
            left=left.data,
            right=right.data,
            check_index_type=check_index_type,
            obj=obj,
            **kwargs,
        )
    except AssertionError as e:
        raise AssertionError(e.args[0].replace('MultiIndex level', f'{obj}.index node'))


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
        obj = 'Graph.nodes.data'
    assert_nodes_equal(left=left.nodes, right=right.nodes, obj=obj, **kwargs)

    if not flag_obj:
        obj = 'Graph.edges.data'
    assert_edges_equal(left=left.edges, right=right.edges, obj=obj, **kwargs)

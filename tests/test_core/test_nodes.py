from functools import partial

import pytest
import pandas as pd

import tahini.core.nodes
import tahini.testing

assert_frame_equal = partial(
    pd.testing.assert_frame_equal,
    check_dtype=False,
    check_column_type=False,
    check_index_type=False,
)


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict()),
    # order
    ([], dict(order=1)),
])
def test_nodes_init_simple(args, kwargs):
    tahini.core.nodes.Nodes(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(), tahini.core.nodes.Nodes()),
    # order
    ([], dict(order=1), tahini.core.nodes.Nodes(index=[0])),
    ([], dict(order=2), tahini.core.nodes.Nodes(index=[0, 1])),
])
def test_nodes_init(args, kwargs, expected):
    nodes = tahini.core.nodes.Nodes(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(nodes, expected)


@pytest.mark.parametrize('nodes, args, kwargs, expected', [
    # empty
    (
        tahini.core.nodes.Nodes(),
        [],
        dict(),
        pd.DataFrame(columns=['position_dim_0', 'position_dim_1'], index=pd.Index([], name='node')),
    ),
])
def test_nodes_get_positions(nodes, args, kwargs, expected):
    positions = nodes.get_positions(*args, **kwargs)
    assert_frame_equal(positions, expected)

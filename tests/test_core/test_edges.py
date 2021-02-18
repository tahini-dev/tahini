from functools import partial

import pytest
import pandas as pd

import tahini.core.edges
import tahini.core.nodes
import tahini.testing

name_nodes = tahini.core.nodes.Nodes().names_index[0]
names_edges = tahini.core.edges.Edges().names_index
names_undirected_edges = tahini.core.edges.UndirectedEdges().names_index

assert_frame_equal = partial(
    pd.testing.assert_frame_equal,
    check_dtype=False,
    check_column_type=False,
    check_index_type=False,
)


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict()),
    # non empty
    ([], dict(index=[(0, 1)])),
])
def test_edges_init_simple(args, kwargs):
    tahini.core.edges.Edges(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict()),
    # non empty
    ([], dict(index=[(0, 1)])),
])
def test_undirected_edges_init_simple(args, kwargs):
    tahini.core.edges.UndirectedEdges(*args, **kwargs)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.edges.Edges(), [], dict(), tahini.core.nodes.Nodes()),
    (tahini.core.edges.UndirectedEdges(), [], dict(), tahini.core.nodes.Nodes()),
    # non empty
    (tahini.core.edges.Edges(index=[(0, 1)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1])),
    (tahini.core.edges.Edges(index=[(0, 1), (0, 2)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1, 2])),
    (tahini.core.edges.UndirectedEdges(index=[(0, 1)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1])),
    (tahini.core.edges.UndirectedEdges(index=[(0, 1), (0, 2)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 1, 2])),
    # order matters
    (tahini.core.edges.Edges(index=[(0, 2), (0, 1)]), [], dict(), tahini.core.nodes.Nodes(index=[0, 2, 1])),
])
def test_edges_nodes(edges, args, kwargs, expected):
    nodes = edges.nodes
    tahini.testing.testing.assert_container_equal(nodes, expected)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (tahini.core.edges.Edges(), [], dict(), tahini.core.edges.Edges()),
    (tahini.core.edges.UndirectedEdges(), [], dict(), tahini.core.edges.UndirectedEdges()),
    # non empty inputs
    (tahini.core.edges.Edges(), [], dict(nodes=[0]), tahini.core.edges.Edges()),
    (tahini.core.edges.UndirectedEdges(), [], dict(nodes=[0]), tahini.core.edges.UndirectedEdges()),
    # non empty
    (tahini.core.edges.Edges(index=[(0, 1)]), [], dict(nodes=[0]), tahini.core.edges.Edges()),
    (tahini.core.edges.UndirectedEdges(index=[(0, 1)]), [], dict(nodes=[0]), tahini.core.edges.UndirectedEdges()),
    (tahini.core.edges.Edges(index=[(0, 1)]), [], dict(nodes=[0, 1]), tahini.core.edges.Edges(index=[(0, 1)])),
    (
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
        [],
        dict(nodes=[0, 1]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
    ),
    (tahini.core.edges.Edges(index=[(0, 1), (0, 2)]), [], dict(nodes=[0, 1]), tahini.core.edges.Edges(index=[(0, 1)])),
    (
        tahini.core.edges.UndirectedEdges(index=[(0, 1), (0, 2)]),
        [],
        dict(nodes=[0, 1]),
        tahini.core.edges.UndirectedEdges(index=[(0, 1)]),
    ),
])
def test_edges_keep_nodes(edges, args, kwargs, expected):
    edges_keep_nodes = edges.keep_nodes(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(edges_keep_nodes, expected)


@pytest.mark.parametrize('edges, args, kwargs, expected', [
    # empty
    (
        tahini.core.edges.Edges(),
        [],
        dict(positions_nodes=tahini.core.nodes.Nodes().get_positions()),
        tahini.core.edges.Edges(data=pd.DataFrame(
            columns=['position_dim_0_start', 'position_dim_1_start', 'position_dim_0_end', 'position_dim_1_end'],
        )).data,
    ),
    (
        tahini.core.edges.UndirectedEdges(),
        [],
        dict(positions_nodes=tahini.core.nodes.Nodes().get_positions()),
        tahini.core.edges.UndirectedEdges(data=pd.DataFrame(
            columns=['position_dim_0_start', 'position_dim_1_start', 'position_dim_0_end', 'position_dim_1_end'],
        )).data,
    ),
    # non empty
    (
        tahini.core.edges.Edges(index=[(0, 1)]),
        [],
        dict(positions_nodes=tahini.core.nodes.Nodes(index=[0, 1]).get_positions()),
        tahini.core.edges.Edges(
            index=[(0, 1)],
            data=pd.DataFrame(data=dict(
                position_dim_0_start=[1],
                position_dim_1_start=[0],
                position_dim_0_end=[-1],
                position_dim_1_end=[0],
            )),
        ).data,
    ),
])
def test_edges_get_positions(edges, args, kwargs, expected):
    df = edges.get_positions(*args, **kwargs)
    assert_frame_equal(df, expected)

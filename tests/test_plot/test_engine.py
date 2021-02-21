from functools import partial

import pytest
import pandas as pd
import plotly.graph_objects as go

import tahini.plot.engine
import tahini.core.graph
import tahini.testing

name_nodes = tahini.core.graph.Graph().nodes.names_index[0]
names_edges = tahini.core.graph.Graph().edges.names_index
names_undirected_edges = tahini.core.graph.UndirectedGraph().edges.names_index

assert_frame_equal = partial(
    pd.testing.assert_frame_equal,
    check_dtype=False,
    check_column_type=False,
    check_index_type=False,
)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty graph
    ([], dict(graph=tahini.core.graph.Graph()), tahini.core.graph.Graph())
])
def test_base_init(args, kwargs, expected):
    engine = tahini.plot.engine.Base(*args, **kwargs)
    tahini.testing.assert_graph_equal(engine.graph, expected)


@pytest.mark.parametrize('engine, args, kwargs, expected', [
    # empty graph
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph()),
        [],
        dict(),
        pd.DataFrame(columns=[name_nodes, 'position_dim_0', 'position_dim_1']),
    ),
    # columns in graph
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph(nodes_data=dict(a=[]))),
        [],
        dict(),
        pd.DataFrame(columns=[name_nodes, 'a', 'position_dim_0', 'position_dim_1']),
    ),
    # input nodes positions
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph()),
        [],
        dict(positions_nodes=pd.DataFrame(columns=[name_nodes, 'position_dim_0', 'position_dim_1'])),
        pd.DataFrame(columns=[name_nodes, 'position_dim_0', 'position_dim_1']),
    ),
    # input nodes positions with index
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph()),
        [],
        dict(
            positions_nodes=pd.DataFrame(
                columns=['position_dim_0', 'position_dim_1'],
                index=pd.Index([], name=name_nodes),
            ),
        ),
        pd.DataFrame(columns=[name_nodes, 'position_dim_0', 'position_dim_1']),
    ),
])
def test_get_positions_nodes(engine, args, kwargs, expected):
    df = engine.get_positions_nodes(*args, **kwargs)
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('engine, args, kwargs, expected', [
    # empty graph
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph()),
        [],
        dict(),
        pd.DataFrame(columns=[
            *names_edges,
            'position_dim_0_end',
            'position_dim_0_start',
            'position_dim_1_end',
            'position_dim_1_start',
        ]),
    ),
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.UndirectedGraph()),
        [],
        dict(),
        pd.DataFrame(columns=[
            *names_undirected_edges,
            'position_dim_0_end',
            'position_dim_0_start',
            'position_dim_1_end',
            'position_dim_1_start',
        ]),
    ),
    # columns in graph
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph(edges_data=dict(a=[]))),
        [],
        dict(),
        pd.DataFrame(columns=[
            *names_edges,
            'a',
            'position_dim_0_end',
            'position_dim_0_start',
            'position_dim_1_end',
            'position_dim_1_start',
        ]),
    ),
    # inputs positions_edges
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph()),
        [],
        dict(
            positions_edges=pd.DataFrame(columns=[
                *names_edges,
                'position_dim_0_end',
                'position_dim_0_start',
                'position_dim_1_end',
                'position_dim_1_start',
            ]),
        ),
        pd.DataFrame(columns=[
            *names_edges,
            'position_dim_0_end',
            'position_dim_0_start',
            'position_dim_1_end',
            'position_dim_1_start',
        ]),
    ),
    # inputs positions_edges with index
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph()),
        [],
        dict(
            positions_edges=pd.DataFrame(
                columns=['position_dim_0_end', 'position_dim_0_start', 'position_dim_1_end', 'position_dim_1_start'],
                index=tahini.core.graph.Graph().edges.data.index,
            ),
        ),
        pd.DataFrame(columns=[
            *names_edges,
            'position_dim_0_end',
            'position_dim_0_start',
            'position_dim_1_end',
            'position_dim_1_start',
        ]),
    ),
    # non empty
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph.star(order=3)),
        [],
        dict(),
        pd.DataFrame(data={
            names_edges[0]: [0, 0],
            names_edges[1]: [1, 2],
            'position_dim_0_end': [-0.5, -0.5],
            'position_dim_0_start': [1, 1],
            'position_dim_1_end': [0.8660254037844387, -0.8660254037844385],
            'position_dim_1_start': [0, 0],
        }),
    ),
])
def test_get_positions_edges(engine, args, kwargs, expected):
    df = engine.get_positions_edges(*args, **kwargs)
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('engine, args, kwargs', [
    # empty
    (tahini.plot.engine.Plotly(graph=tahini.core.graph.Graph()), [], dict()),
    # non empty with nodes
    (tahini.plot.engine.Plotly(graph=tahini.core.graph.Graph(order=10)), [], dict()),
    # non empty with nodes and edges
    (tahini.plot.engine.Plotly(graph=tahini.core.graph.Graph.star(order=10)), [], dict()),
    (tahini.plot.engine.Plotly(graph=tahini.core.graph.UndirectedGraph.star(order=10)), [], dict()),
    # non default nodes
    (
        tahini.plot.engine.Plotly(graph=tahini.core.graph.Graph(nodes=['a', 'b', 'c'], edges=[('a', 'b'), ('b', 'c')])),
        [],
        dict(),
    ),
])
def test_plotly_plot(engine, args, kwargs):
    fig = engine.plot(*args, **kwargs)
    # fig.show()
    assert isinstance(fig, go.Figure)

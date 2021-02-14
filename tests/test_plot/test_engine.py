from functools import partial

import pytest
import pandas as pd
import plotly.graph_objects as go

import tahini.plot.engine
import tahini.core.graph
import tahini.testing

name_nodes = tahini.core.graph.Graph().nodes.names_index[0]

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
        pd.DataFrame(columns=['position_dim_0', 'position_dim_1'], index=pd.Index([], name=name_nodes)),
    ),
    # columns in graph
    (
        tahini.plot.engine.Base(graph=tahini.core.graph.Graph(nodes_data=dict(a=[]))),
        [],
        dict(),
        pd.DataFrame(columns=['a', 'position_dim_0', 'position_dim_1'], index=pd.Index([], name=name_nodes)),
    ),
])
def test_base_get_data_frame_plot(engine, args, kwargs, expected):
    df = engine.get_data_frame_plot(*args, **kwargs)
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('engine, args, kwargs', [
    # non empty graph
    (tahini.plot.engine.Plotly(graph=tahini.core.graph.Graph(order=10)), [], dict()),
    # empty
    (tahini.plot.engine.Plotly(graph=tahini.core.graph.Graph()), [], dict()),
])
def test_plot(engine, args, kwargs):
    fig = engine.plot(*args, **kwargs)
    assert isinstance(fig, go.Figure)

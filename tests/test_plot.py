import pytest

import pandas as pd
import numpy as np

from tahini.core import Graph
import tahini.plot


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # mismatch between dim and center length
    ([], dict(center=[0, 0], dim=1), ValueError, "Shape of 'center' has to be ('dim',)"),
    ([], dict(center=[0], dim=2), ValueError, "Shape of 'center' has to be ('dim',)"),
    ([], dict(center=[0]), ValueError, "Shape of 'center' has to be ('dim',)"),
])
def test__process_parameters_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.plot._process_parameters(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, center_expected, dim_expected', [
    # empty
    ([], dict(), np.zeros(2), 2),
    # dim
    ([], dict(dim=1), np.zeros(1), 1),
    ([], dict(dim=2), np.zeros(2), 2),
    # center and dim
    ([], dict(center=[1], dim=1), np.ones(1), 1),
    # center
    ([], dict(center=[1, 1]), np.ones(2), 2),
])
def test__process_parameters(args, kwargs, center_expected, dim_expected):
    center, dim = tahini.plot._process_parameters(*args, **kwargs)
    np.testing.assert_array_equal(center, center_expected)
    assert dim == dim_expected


def get_data_frame(*args, nodes=None, dim=None, **kwargs):
    if nodes is None:
        nodes = []
    if dim is None:
        dim = 2
    index = Graph(nodes=nodes).nodes.data.index
    df = pd.DataFrame(*args, index=index, columns=[f'position_dim_{i}' for i in range(dim)], **kwargs).astype('float64')
    return df


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty graph
    ([], dict(array=np.zeros((0, 2)), graph=Graph(), dim=2), get_data_frame(data=np.zeros((0, 2)))),
    # non empty graph
    ([], dict(array=np.zeros((1, 2)), graph=Graph(nodes=[0]), dim=2), get_data_frame(data=np.zeros((1, 2)), nodes=[0])),
    ([], dict(array=np.ones((1, 2)), graph=Graph(nodes=[0]), dim=2), get_data_frame(data=np.ones((1, 2)), nodes=[0])),
    # non default dim
    ([], dict(array=np.ones((0, 0)), graph=Graph(), dim=0), get_data_frame(data=np.zeros((0, 0)), dim=0)),
    ([], dict(array=np.ones((0, 1)), graph=Graph(), dim=1), get_data_frame(data=np.zeros((0, 1)), dim=1)),
    (
        [],
        dict(array=np.zeros((1, 1)), graph=Graph(nodes=[0]), dim=1),
        get_data_frame(data=np.zeros((1, 1)), nodes=[0], dim=1),
    ),
    (
        [],
        dict(array=np.zeros((1, 3)), graph=Graph(nodes=[0]), dim=3),
        get_data_frame(data=np.zeros((1, 3)), nodes=[0], dim=3),
    ),
    (
        [],
        dict(array=np.zeros((2, 3)), graph=Graph(nodes=[0, 1]), dim=3),
        get_data_frame(data=np.zeros((2, 3)), nodes=[0, 1], dim=3),
    ),
])
def test__array_to_data_frame(args, kwargs, expected):
    df = tahini.plot._array_to_data_frame(*args, **kwargs)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    ([], dict(graph=Graph(), dim=1), ValueError, "'dim' has to be > 1 for circular layout for graph positions"),
])
def test__get_positions_layout_circular_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.plot._get_positions_layout_circular(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty graph
    ([], dict(graph=Graph(), dim=2), np.zeros((0, 2))),
    # empty graph non one scale
    ([], dict(graph=Graph(), dim=2, scale=2), np.zeros((0, 2))),
    # graph with order 1
    ([], dict(graph=Graph(order=1), dim=2), np.zeros((1, 2))),
    # graph with order 2
    ([], dict(graph=Graph(order=2), dim=2), np.array([[1, 0], [-1, 0]])),
    # graph with order 2 and non one scale
    ([], dict(graph=Graph(order=2), dim=2, scale=2), np.array([[2, 0], [-2, 0]])),
    # graph with order 2 and dim > 2
    ([], dict(graph=Graph(order=2), dim=3), np.array([[1, 0, 0], [-1, 0, 0]])),
])
def test__get_positions_layout_circular(args, kwargs, expected):
    positions = tahini.plot._get_positions_layout_circular(*args, **kwargs)
    np.testing.assert_array_almost_equal(positions, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty graph
    ([], dict(graph=Graph(), dim=2), np.zeros((0, 2))),
    # empty graph with seed
    ([], dict(graph=Graph(), dim=2, seed=0), np.zeros((0, 2))),
    # graph of order 1
    ([], dict(graph=Graph(order=1), dim=2, seed=0), np.array([[0.548814, 0.715189]])),
    # graph of order 2
    ([], dict(graph=Graph(order=2), dim=2, seed=0), np.array([[0.548814, 0.715189], [0.602763, 0.544883]])),
    # graph of order 2 and dim 3
    (
        [],
        dict(graph=Graph(order=2), dim=3, seed=0),
        np.array([[0.548814, 0.715189, 0.602763], [0.544883, 0.423655, 0.645894]]),
    ),
])
def test__get_positions_layout_random(args, kwargs, expected):
    positions = tahini.plot._get_positions_layout_random(*args, **kwargs)
    np.testing.assert_array_almost_equal(positions, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty graph
    ([], dict(graph=Graph()), get_data_frame()),
    # non default dim
    ([], dict(graph=Graph(), dim=3), get_data_frame(dim=3)),
    # graph of order 1
    ([], dict(graph=Graph(order=1)), get_data_frame(data=[[0, 0]], nodes=range(1))),
    # graph of order 2
    ([], dict(graph=Graph(order=2)), get_data_frame(data=[[1, 0], [-1, 0]], nodes=range(2))),
    # layout - circular
    (
        [],
        dict(graph=Graph(order=2), layout='circular'),
        get_data_frame(data=[[1, 0], [-1, 0]], nodes=range(2)),
    ),
    # layout - random
    (
        [],
        dict(graph=Graph(order=2), layout='random', seed=0),
        get_data_frame(data=[[0.548814, 0.715189], [0.602763, 0.544883]], nodes=range(2)),
    ),
    # center
    ([], dict(graph=Graph(order=1), center=[1, 1]), get_data_frame(data=[[1, 1]], nodes=range(1))),
])
def test_get_positions(args, kwargs, expected):
    positions = tahini.plot.get_positions(*args, **kwargs)
    pd.testing.assert_frame_equal(positions, expected)

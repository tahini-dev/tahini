import pytest

import pandas as pd
import numpy as np

import tahini.plot.positions
import tahini.core.nodes


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # mismatch between dim and center length
    ([], dict(center=[0, 0], dim=1), ValueError, "Shape of 'center' has to be ('dim',)"),
    ([], dict(center=[0], dim=2), ValueError, "Shape of 'center' has to be ('dim',)"),
    ([], dict(center=[0]), ValueError, "Shape of 'center' has to be ('dim',)"),
])
def test__process_parameters_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.plot.positions._process_parameters(*args, **kwargs)
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
    center, dim = tahini.plot.positions._process_parameters(*args, **kwargs)
    np.testing.assert_array_equal(center, center_expected)
    assert dim == dim_expected


def get_data_frame(*args, items=None, dim=None, **kwargs):
    if items is None:
        items = []
    if dim is None:
        dim = 2
    df = pd.DataFrame(*args, index=items, columns=[f'position_dim_{i}' for i in range(dim)], **kwargs).astype('float64')
    return df


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty items
    ([], dict(array=np.zeros((0, 2)), items=[], dim=2), get_data_frame(data=np.zeros((0, 2)))),
    # non empty items
    ([], dict(array=np.zeros((1, 2)), items=[0], dim=2), get_data_frame(data=np.zeros((1, 2)), items=[0])),
    ([], dict(array=np.ones((1, 2)), items=[0], dim=2), get_data_frame(data=np.ones((1, 2)), items=[0])),
    # non default dim
    ([], dict(array=np.ones((0, 0)), items=[], dim=0), get_data_frame(data=np.zeros((0, 0)), dim=0)),
    ([], dict(array=np.ones((0, 1)), items=[], dim=1), get_data_frame(data=np.zeros((0, 1)), dim=1)),
    (
        [],
        dict(array=np.zeros((1, 1)), items=[0], dim=1),
        get_data_frame(data=np.zeros((1, 1)), items=[0], dim=1),
    ),
    (
        [],
        dict(array=np.zeros((1, 3)), items=[0], dim=3),
        get_data_frame(data=np.zeros((1, 3)), items=[0], dim=3),
    ),
    (
        [],
        dict(array=np.zeros((2, 3)), items=[0, 1], dim=3),
        get_data_frame(data=np.zeros((2, 3)), items=[0, 1], dim=3),
    ),
])
def test__array_to_data_frame(args, kwargs, expected):
    df = tahini.plot.positions._array_to_data_frame(*args, **kwargs)
    pd.testing.assert_frame_equal(df, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    ([], dict(items=[], dim=1), ValueError, "'dim' has to be > 1 for circular layout"),
])
def test__get_circular_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.plot.positions._get_circular(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty items
    ([], dict(items=[], dim=2), np.zeros((0, 2))),
    # empty items non one scale
    ([], dict(items=[], dim=2, scale=2), np.zeros((0, 2))),
    # items with order 1
    ([], dict(items=[0], dim=2), np.zeros((1, 2))),
    # items with order 2
    ([], dict(items=[0, 1], dim=2), np.array([[1, 0], [-1, 0]])),
    # items with order 2 and non one scale
    ([], dict(items=[0, 1], dim=2, scale=2), np.array([[2, 0], [-2, 0]])),
    # items with order 2 and dim > 2
    ([], dict(items=[0, 1], dim=3), np.array([[1, 0, 0], [-1, 0, 0]])),
])
def test__get_circular(args, kwargs, expected):
    positions = tahini.plot.positions._get_circular(*args, **kwargs)
    np.testing.assert_array_almost_equal(positions, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty items
    ([], dict(items=[], dim=2), np.zeros((0, 2))),
    # empty items with seed
    ([], dict(items=[], dim=2, seed=0), np.zeros((0, 2))),
    # items of order 1
    ([], dict(items=[0], dim=2, seed=0), np.array([[0.548814, 0.715189]])),
    # items of order 2
    ([], dict(items=[0, 1], dim=2, seed=0), np.array([[0.548814, 0.715189], [0.602763, 0.544883]])),
    # items of order 2 and dim 3
    (
        [],
        dict(items=[0, 1], dim=3, seed=0),
        np.array([[0.548814, 0.715189, 0.602763], [0.544883, 0.423655, 0.645894]]),
    ),
])
def test__get_random(args, kwargs, expected):
    positions = tahini.plot.positions._get_random(*args, **kwargs)
    np.testing.assert_array_almost_equal(positions, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty items
    ([], dict(items=[]), get_data_frame()),
    # non default dim
    ([], dict(items=[], dim=3), get_data_frame(dim=3)),
    # items of order 1
    ([], dict(items=[0]), get_data_frame(data=[[0, 0]], items=range(1))),
    # items of order 2
    ([], dict(items=[0, 1]), get_data_frame(data=[[1, 0], [-1, 0]], items=range(2))),
    # layout - circular
    (
        [],
        dict(items=[0, 1], layout='circular'),
        get_data_frame(data=[[1, 0], [-1, 0]], items=range(2)),
    ),
    # layout - random
    (
        [],
        dict(items=[0, 1], layout='random', seed=0),
        get_data_frame(data=[[0.548814, 0.715189], [0.602763, 0.544883]], items=range(2)),
    ),
    # center
    ([], dict(items=[0], center=[1, 1]), get_data_frame(data=[[1, 1]], items=range(1))),
])
def test_get(args, kwargs, expected):
    positions = tahini.plot.positions.get(*args, **kwargs)
    pd.testing.assert_frame_equal(positions, expected)

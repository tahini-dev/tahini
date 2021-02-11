from collections.abc import Sequence
from functools import partial
from math import isnan, nan

import pytest
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.pandas import indexes, columns, data_frames
import pandas as pd

import tahini.container
import tahini.testing

names_index_container_data_indexed = 'index'
name_index_internal = 'index_internal'
names_index_container_data_indexed_multi = ('index_0', 'index_1')


def get_data_frame(*args, name_index=names_index_container_data_indexed, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(*args, **kwargs).rename_axis(index=name_index)


def get_data_frame_internal(
        *args,
        index_internal=None,
        name_index=names_index_container_data_indexed,
        **kwargs,
) -> pd.DataFrame:
    df = pd.DataFrame(*args, **kwargs).rename_axis(index=name_index).reset_index()
    if index_internal is None:
        index_internal = df[name_index]
    df.index = pd.Index(index_internal, name=name_index_internal)
    return df


def get_data_frame_index_multi(
        *args,
        names_index=names_index_container_data_indexed_multi,
        index=None,
        **kwargs,
) -> pd.DataFrame:
    if index is None:
        index = pd.MultiIndex(levels=[[]] * len(names_index), codes=[[]] * len(names_index), names=names_index)
    else:
        index = pd.MultiIndex.from_tuples(index, names=names_index)
    return pd.DataFrame(*args, index=index, **kwargs)


def get_data_frame_internal_index_multi(
        *args,
        index_internal=None,
        mapper=None,
        **kwargs,
) -> pd.DataFrame:

    df = get_data_frame_index_multi(*args, **kwargs)

    if mapper is None:
        def identity(x): return x
        mapper = identity

    if index_internal is None:
        index_internal = df.index.to_flat_index().map(mapper)

    df = df.reset_index()
    df.index = pd.Index(index_internal, name=name_index_internal)

    return df


def get_data_frame_internal_simple_index_multi(*arg, **kwargs):
    df = (
        get_data_frame_internal_index_multi(*arg, **kwargs)
        .drop(columns=list(names_index_container_data_indexed_multi))
    )
    return df


get_data_frame_internal_index_multi_sets = partial(get_data_frame_internal_index_multi, mapper=frozenset)

get_data_frame_internal_simple_index_multi_sets = partial(
    get_data_frame_internal_simple_index_multi,
    mapper=frozenset,
)

assert_frame_equal = partial(
    pd.testing.assert_frame_equal,
    check_dtype=False,
    check_column_type=False,
    check_index_type=False,
)

assert_index_equal = partial(pd.testing.assert_index_equal, exact=False)


def check_nan(x):
    try:
        tf = isnan(x)
    except TypeError:
        tf = False
    return tf


@pytest.mark.parametrize('klass', [
    tahini.container.ContainerDataIndexed,
    tahini.container.ContainerDataIndexedMulti,
    tahini.container.ContainerDataIndexedMultiSets,
])
def test_container_data_indexed__names_index(klass):
    assert isinstance(klass._names_index, Sequence)


@pytest.mark.parametrize('klass', [
    tahini.container.ContainerDataIndexed,
    tahini.container.ContainerDataIndexedMulti,
    tahini.container.ContainerDataIndexedMultiSets,
])
def test_container_data_indexed__name_index_internal(klass):
    assert isinstance(klass._name_index_internal, str)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty index
    ([], dict(index=pd.Index([])), pd.Index([], name=name_index_internal)),
    # non empty index
    ([], dict(index=pd.Index([0])), pd.Index([0], name=name_index_internal)),
    # empty multi index
    ([], dict(index=pd.MultiIndex.from_arrays([[]])), pd.MultiIndex.from_arrays([[]]).rename(name_index_internal)),
])
def test_container_data_indexed__create_index_internal(args, kwargs, expected):
    index = tahini.container.ContainerDataIndexed._create_index_internal(*args, **kwargs)
    assert_index_equal(index, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # non unique index
    ([], dict(index=pd.Index([0, 0])), ValueError, "Index needs to be unique for 'ContainerDataIndexed'"),
])
def test_container_data_indexed__validate_index_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.container.ContainerDataIndexed._validate_index(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(), get_data_frame_internal()),
    # non empty index
    ([], dict(index=[0]), get_data_frame_internal(index=[0])),
    # empty index
    ([], dict(index=[]), get_data_frame_internal()),
    # empty container idempotent
    ([], dict(index=tahini.container.ContainerDataIndexed()), get_data_frame_internal()),
    # empty data dict
    ([], dict(data=dict()), get_data_frame_internal()),
    # empty data records
    ([], dict(data=[]), get_data_frame_internal()),
    # empty data frame
    ([], dict(data=pd.DataFrame()), get_data_frame_internal()),
    # data dict
    ([], dict(data=dict(a=[1])), get_data_frame_internal(data=dict(a=[1]))),
    # dict and index
    ([], dict(data=dict(a=[1]), index=['z']), get_data_frame_internal(data=dict(a=[1]), index=['z'])),
    # data frame
    ([], dict(data=pd.DataFrame(data=dict(a=[1]))), get_data_frame_internal(data=dict(a=[1]))),
    # data frame with index
    (
        [],
        dict(data=pd.DataFrame(data=dict(a=[1]), index=['z'])),
        get_data_frame_internal(data=dict(a=[1]), index=['z']),
    ),
    # data frame and index
    (
        [],
        dict(data=pd.DataFrame(data=dict(a=[1])), index=['z']),
        get_data_frame_internal(data=dict(a=[1]), index=['z']),
    ),
    # data records
    ([], dict(data=[[1]]), get_data_frame_internal(data=[[1]])),
    ([], dict(data=['a', 'b']), get_data_frame_internal({0: ['a', 'b']})),
    ([], dict(data=[['a'], ['b']]), get_data_frame_internal({0: ['a', 'b']})),
    ([], dict(data=[['a', 'b']]), get_data_frame_internal({0: ['a'], 1: ['b']})),
    # container idempotent
    (
        [],
        dict(index=tahini.container.ContainerDataIndexed(data=pd.DataFrame(data=dict(a=[1]), index=['z']))),
        get_data_frame_internal(data=dict(a=[1]), index=['z']),
    ),
    # index as column
    ([], dict(data=dict(index=[0, 1])), get_data_frame_internal(index=[0, 1])),
])
def test_container_data_indexed_init(args, kwargs, expected):
    container = tahini.container.ContainerDataIndexed(*args, **kwargs)
    assert_frame_equal(container.data_internal, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(index=pd.Index([])), pd.Index([], name=names_index_container_data_indexed)),
    # non empty
    ([], dict(index=pd.Index([0])), pd.Index([0], name=names_index_container_data_indexed)),
])
def test_container_data_indexed__validate_index(args, kwargs, expected):
    index = tahini.container.ContainerDataIndexed._validate_index(*args, **kwargs)
    assert_index_equal(index, expected)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(data=pd.DataFrame()), get_data_frame()),
    # non empty index
    ([], dict(data=pd.DataFrame(index=['a', 'b'])), get_data_frame(index=['a', 'b'])),
    # non empty index with name
    (
        [],
        dict(data=pd.DataFrame(index=pd.Index(['a', 'b'], name=f'not_{names_index_container_data_indexed}'))),
        get_data_frame(index=['a', 'b']),
    ),
    # non empty data
    ([], dict(data=pd.DataFrame(data=dict(a=[0, 1], b=[0, 1]))), get_data_frame(data=dict(a=[0, 1], b=[0, 1]))),
])
def test_container_data_indexed__validate_data(args, kwargs, expected):
    df = tahini.container.ContainerDataIndexed._validate_data(*args, **kwargs)
    assert_frame_equal(df, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # non unique index
    ([], dict(index=[0, 0]), ValueError, "Index needs to be unique for 'ContainerDataIndexed'"),
    # non matching length between index and data
    (
        [],
        dict(data=pd.DataFrame(data=dict(a=[1])), index=[0, 1]),
        ValueError,
        "Length mismatch: Expected axis has 1 elements, new values have 2 elements",
    ),
    # non matching length between index and data
    (
        [],
        dict(data=pd.DataFrame(data=dict(a=[1, 2])), index=[0]),
        ValueError,
        "Length mismatch: Expected axis has 2 elements, new values have 1 elements",
    ),
])
def test_container_data_indexed_init_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.container.ContainerDataIndexed(*args, **kwargs)
    assert e.value.args[0] == message_error


types_index = (
    st.iterables,
    indexes,
)

elements_non_specific = (
    st.binary,
    st.booleans,
    st.characters,
    st.complex_numbers,
    st.dates,
    st.datetimes,
    st.fractions,
    st.integers,
    st.none,
    st.randoms,
    st.text,
    st.times,
    st.uuids,
)

elements_specific = (
    # pandas.Timedeltas max and min do not match python standard library datetime.timedelta max and min
    (
        st.timedeltas,
        dict(min_value=pd.Timedelta.min.to_pytimedelta(), max_value=pd.Timedelta.max.to_pytimedelta()),
        lambda x: True,
        lambda x: True,
    ),
    # error with decimals not being able to hash snan
    (
        st.decimals,
        dict(),
        lambda x: not x.is_snan(),
        lambda x: True,
    ),
    # cannot have duplicate nans
    (
        st.floats,
        dict(),
        lambda x: True,
        lambda container: sum([isnan(item) for item in container]) < 2,
    ),
)


@pytest.mark.parametrize('type_index', types_index)
@pytest.mark.parametrize('elements, kwargs_elements, filter_elements, filter_type_index', [
    *((item, dict(), lambda x: True, lambda x: True) for item in elements_non_specific),
    *elements_specific,
])
@given(data=st.data())
def test_container_data_indexed_init_index_single_elements_type(
        type_index,
        elements,
        kwargs_elements,
        filter_elements,
        filter_type_index,
        data,
):
    container = tahini.container.ContainerDataIndexed(
        index=data.draw(
            type_index(
                elements=elements(**kwargs_elements).filter(filter_elements),
                unique=True,
            )
            .filter(filter_type_index)
        ),
    )
    assert isinstance(container.data_internal, pd.DataFrame)


@pytest.mark.parametrize('type_index', types_index)
@pytest.mark.parametrize('elements', [
    pytest.param(
        st.timedeltas,
        marks=pytest.mark.xfail(
            reason='pandas.Timedeltas max and min do not match python standard library datetime.timedelta max and min',
        ),
    ),
    pytest.param(
        st.decimals,
        marks=pytest.mark.xfail(
            reason='error with decimals not being able to hash snan',
        ),
    ),
    pytest.param(
        st.floats,
        marks=pytest.mark.xfail(
            reason='error with duplicate nans',
        ),
    ),
])
@given(data=st.data())
def test_container_data_indexed_init_index_single_elements_type_xfail(type_index, elements, data):
    container = tahini.container.ContainerDataIndexed(
        index=data.draw(type_index(elements=elements.param(), unique=True)),
    )


@pytest.fixture(scope='module')
def list_elements():
    output_value = (
        *(elements() for elements in elements_non_specific),
        *(item[0](**item[1]).filter(item[2]) for item in elements_specific),
    )
    return output_value


@pytest.mark.parametrize('type_index', types_index)
@given(data=st.data())
def test_container_data_indexed_init_index_multiple_elements_type(type_index, list_elements, data):
    index = data.draw(
        type_index(elements=st.one_of(*list_elements), unique=True)
        .filter(lambda x: sum([check_nan(item) for item in x]) < 2)
    )
    container = tahini.container.ContainerDataIndexed(index=index)
    assert isinstance(container.data_internal, pd.DataFrame)


@given(data=data_frames(columns=(columns('A', elements=st.integers())), index=indexes(elements=st.integers())))
def test_container_data_indexed_init_data_data_frame(data):
    container = tahini.container.ContainerDataIndexed(data=data)
    assert isinstance(container.data_internal, pd.DataFrame)


@pytest.mark.parametrize('container, data, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), pd.DataFrame(), get_data_frame()),
    # non empty container
    (tahini.container.ContainerDataIndexed(data=dict(a=['1'])), pd.DataFrame(), get_data_frame()),
    # empty container and non empty data
    (tahini.container.ContainerDataIndexed(), pd.DataFrame(data=dict(a=['1'])), get_data_frame(data=dict(a=['1']))),
    # non empty container and data
    (
        tahini.container.ContainerDataIndexed(data=dict(a=['1'])),
        pd.DataFrame(data=dict(b=[2])),
        get_data_frame(data=dict(b=[2])),
    ),
])
def test_container_data_indexed_data(container, data, expected):
    container.data = data
    assert_frame_equal(container.data, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), get_data_frame_internal()),
    # non empty container
    (
        tahini.container.ContainerDataIndexed(data=dict(a=['1'])),
        get_data_frame_internal(data=dict(a=['1'])),
    ),
])
def test_container_data_indexed_data_internal(container, expected):
    assert_frame_equal(container.data_internal, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), get_data_frame(name_index=name_index_internal)),
    # non empty container
    (
        tahini.container.ContainerDataIndexed(data=dict(a=['1'])),
        get_data_frame(data=dict(a=['1']), name_index=name_index_internal),
    ),
])
def test_container_data_indexed_data_testing(container, expected):
    assert_frame_equal(container.data_testing, expected)


@pytest.mark.parametrize('container, args, kwargs, type_error, message_error', [
    # index not in container
    (tahini.container.ContainerDataIndexed(), [], dict(index=[0]), KeyError, "[0] not found in axis")
])
def test_container_data_indexed_drop_error(container, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        container.drop(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), [], dict(), tahini.container.ContainerDataIndexed()),
    # empty container ignore error
    (
        tahini.container.ContainerDataIndexed(),
        [],
        dict(index=[0], errors='ignore'),
        tahini.container.ContainerDataIndexed(),
    ),
    # empty inputs
    (tahini.container.ContainerDataIndexed(index=[0]), [], dict(), tahini.container.ContainerDataIndexed(index=[0])),
    # non empty
    (tahini.container.ContainerDataIndexed(index=[0]), [], dict(index=[0]), tahini.container.ContainerDataIndexed()),
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(index=[0, 1]),
        tahini.container.ContainerDataIndexed(),
    ),
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(index=[1]),
        tahini.container.ContainerDataIndexed(index=[0]),
    ),
    # drop columns
    (
        tahini.container.ContainerDataIndexed(index=[0], columns=['a']),
        [],
        dict(columns=['a']),
        tahini.container.ContainerDataIndexed(index=[0]),
    ),
])
def test_container_data_indexed_drop(container, args, kwargs, expected):
    container = container.drop(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), [], dict(), tahini.container.ContainerDataIndexed()),
    # empty inputs
    (tahini.container.ContainerDataIndexed(index=[0]), [], dict(), tahini.container.ContainerDataIndexed(index=[0])),
    # empty container and non empty index
    (tahini.container.ContainerDataIndexed(), [], dict(index=[0]), tahini.container.ContainerDataIndexed(index=[0])),
    # empty container and non empty data
    (
        tahini.container.ContainerDataIndexed(),
        [],
        dict(data=dict(a=[1])),
        tahini.container.ContainerDataIndexed(data=dict(a=[1])),
    ),
    # update with no new changes
    (
        tahini.container.ContainerDataIndexed(index=[0]),
        [],
        dict(index=[0]),
        tahini.container.ContainerDataIndexed(index=[0]),
    ),
    # update seems to sort
    (
        tahini.container.ContainerDataIndexed(index=[0]),
        [],
        dict(index=[2, 1]),
        tahini.container.ContainerDataIndexed(index=[0, 1, 2]),
    ),
    # new column and index
    (
        tahini.container.ContainerDataIndexed(),
        [],
        dict(data=dict(a=[1, 2])),
        tahini.container.ContainerDataIndexed(data=dict(a=[1, 2])),
    ),
    # new column for given index
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(index=[0, 1], data=dict(a=[1, 2])),
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2])),
    ),
    # new column and index item
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(index=[1, 2], data=dict(a=[1, 2])),
        tahini.container.ContainerDataIndexed(index=[0, 1, 2], data=dict(a=[nan, 1, 2])),
    ),
    # cannot update to nan with default func
    (
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2])),
        [],
        dict(index=[1], data=dict(a=[nan])),
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2])),
    ),
    # single value in column
    (
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2])),
        [],
        dict(index=[1], data=dict(a=[3])),
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 3])),
    ),
    # single value in column
    (
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2])),
        [],
        dict(index=[0], data=dict(a=[3])),
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[3, 2])),
    ),
    # new additional column
    (
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2])),
        [],
        dict(index=[0, 1], data=dict(b=[2, 3])),
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2], b=[2, 3])),
    ),
    # row update
    (
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[1, 2], b=[2, 3])),
        [],
        dict(index=[0], data=dict(a=[4], b=[5])),
        tahini.container.ContainerDataIndexed(index=[0, 1], data=dict(a=[4, 2], b=[5, 3])),
    ),
])
def test_container_data_indexed_update(container, args, kwargs, expected):
    container = container.update(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, args, kwargs, type_error, message_error', [
    # missing map multiple
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(mapper={}),
        ValueError,
        "Index needs to be unique for 'ContainerDataIndexed'",
    ),
])
def test_container_data_indexed_map_error(container, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        container.map(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), [], dict(), tahini.container.ContainerDataIndexed()),
    # empty inputs
    (tahini.container.ContainerDataIndexed(index=[0]), [], dict(), tahini.container.ContainerDataIndexed(index=[0])),
    # empty container
    (tahini.container.ContainerDataIndexed(), [], dict(mapper=dict()), tahini.container.ContainerDataIndexed()),
    (tahini.container.ContainerDataIndexed(), [], dict(mapper=dict(a=1)), tahini.container.ContainerDataIndexed()),
    # non empty
    (
        tahini.container.ContainerDataIndexed(index=[0]),
        [],
        dict(mapper={0: 1}), tahini.container.ContainerDataIndexed(index=[1]),
    ),
    # change index type
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(mapper={0: 'a', 1: 'b'}),
        tahini.container.ContainerDataIndexed(index=['a', 'b']),
    ),
    # missing map
    (
        tahini.container.ContainerDataIndexed(index=[0, 1]),
        [],
        dict(mapper={0: 'a'}),
        tahini.container.ContainerDataIndexed(index=['a', nan]),
    ),
])
def test_container_data_indexed_map(container, args, kwargs, expected):
    container = container.map(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, expected', [
    (tahini.container.ContainerDataIndexed(), f'ContainerDataIndexed(index={get_data_frame().index})'),
    (
        tahini.container.ContainerDataIndexedMulti(),
        f'ContainerDataIndexedMulti(index={get_data_frame_index_multi().index})',
    ),
    (
        tahini.container.ContainerDataIndexedMultiSets(),
        f'ContainerDataIndexedMultiSets(index={get_data_frame_index_multi().index})',
    ),
])
def test_container_data_indexed_repr(container, expected):
    repr_container = repr(container)
    assert repr_container == expected


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), []),
    (tahini.container.ContainerDataIndexedMulti(), []),
    (tahini.container.ContainerDataIndexedMultiSets(), []),
    # non empty
    (tahini.container.ContainerDataIndexed(index=[0]), [0]),
    (tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]), [(0, 1), (0, 2)]),
    (tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]), [frozenset((0, 1)), frozenset((0, 2))]),
])
def test_container_data_indexed_iter(container, expected):
    assert [item for item in container.iter()] == expected
    assert [item for item in container] == expected


@pytest.mark.parametrize('container, item, expected', [
    # not in empty
    (tahini.container.ContainerDataIndexed(), 0, False),
    (tahini.container.ContainerDataIndexedMulti(), (0, 1), False),
    (tahini.container.ContainerDataIndexedMultiSets(), (0, 1), False),
    # contains
    (tahini.container.ContainerDataIndexed(index=[0]), 0, True),
    (tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]), (0, 1), True),
    (tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]), (0, 1), True),
    (tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]), (1, 0), True),
    # not contains
    (tahini.container.ContainerDataIndexed(index=[0]), 1, False),
    (tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]), (0, 2), False),
    (tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]), (1, 0), False),
    (tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]), (0, 2), False),
])
def test_container_data_indexed_contains(container, item, expected):
    assert (item in container) == expected


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), 0),
    (tahini.container.ContainerDataIndexedMulti(), 0),
    (tahini.container.ContainerDataIndexedMultiSets(), 0),
    # non empty
    (tahini.container.ContainerDataIndexed(index=[0]), 1),
    (tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]), 1),
    (tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]), 1),
])
def test_container_data_indexed_len(container, expected):
    assert len(container) == expected


@pytest.mark.parametrize('container_left, container_right, expected', [
    # empty
    (tahini.container.ContainerDataIndexed(), tahini.container.ContainerDataIndexed(), True),
    (tahini.container.ContainerDataIndexedMulti(), tahini.container.ContainerDataIndexedMulti(), True),
    (tahini.container.ContainerDataIndexedMultiSets(), tahini.container.ContainerDataIndexedMultiSets(), True),
    # non empty
    (tahini.container.ContainerDataIndexed(index=[0]), tahini.container.ContainerDataIndexed(index=[0]), True),
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        True,
    ),
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        True,
    ),
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        tahini.container.ContainerDataIndexedMultiSets(index=[(1, 0)]),
        True,
    ),
    # empty versus non empty
    (tahini.container.ContainerDataIndexed(), tahini.container.ContainerDataIndexed(index=[0]), False),
    # None right
    (tahini.container.ContainerDataIndexed(), None, False),
    # None left
    (None, tahini.container.ContainerDataIndexed(), False),
    # different order
    (tahini.container.ContainerDataIndexed(index=[1, 2]), tahini.container.ContainerDataIndexed(index=[2, 1]), True),
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 2), (0, 1)]),
        True,
    ),
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 2), (0, 1)]),
        True,
    ),
])
def test_container_data_indexed_eq(container_left, container_right, expected):
    assert (container_left == container_right) == expected


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty multi index
    ([], dict(index=pd.MultiIndex.from_arrays([[]])), pd.Index([], name=name_index_internal)),
    # non empty multi index
    (
        [],
        dict(index=pd.MultiIndex.from_tuples([(0, 1)])),
        pd.Index([(0, 1)]).to_flat_index().rename(name_index_internal),
    ),
])
def test_container_data_indexed_multi__create_index_internal(args, kwargs, expected):
    index = tahini.container.ContainerDataIndexedMulti._create_index_internal(*args, **kwargs)
    assert_index_equal(index, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # non unique index
    (
        [],
        dict(index=pd.Index([(0, 1), (0, 1)])),
        ValueError,
        "Index needs to be unique for 'ContainerDataIndexedMulti'",
    ),
])
def test_container_data_indexed_multi__validate_index_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.container.ContainerDataIndexedMulti._validate_index(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    (
        [],
        dict(index=pd.Index([])),
        pd.MultiIndex(levels=[[]] * 2, codes=[[]] * 2, names=names_index_container_data_indexed_multi),
    ),
    # index
    (
        [],
        dict(index=pd.Index([(0, 1)])),
        pd.MultiIndex.from_tuples([(0, 1)], names=names_index_container_data_indexed_multi),
    ),
    # index
    (
        [],
        dict(index=pd.MultiIndex.from_tuples([(0, 1)])),
        pd.MultiIndex.from_tuples([(0, 1)], names=names_index_container_data_indexed_multi),
    ),
])
def test_container_data_indexed_multi__validate_index(args, kwargs, expected):
    index = tahini.container.ContainerDataIndexedMulti._validate_index(*args, **kwargs)
    assert_index_equal(index, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # non unique
    ([], dict(index=[(0, 1), (0, 1)]), ValueError, "Index needs to be unique for 'ContainerDataIndexedMulti'"),
    # flat index input
    ([], dict(index=[0, 1]), TypeError, "object of type 'int' has no len()"),
    # flat index input
    ([], dict(index=[0, 1, 2]), TypeError, "object of type 'int' has no len()"),
    # flat index input
    ([], dict(index=['a', 'b']), ValueError, "Length of names must match number of levels in MultiIndex."),
])
def test_container_data_indexed_multi_init_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.container.ContainerDataIndexedMulti(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(), get_data_frame_internal_index_multi()),
    # non empty
    ([], dict(index=[(0, 1)]), get_data_frame_internal_index_multi(index=[(0, 1)])),
    # list of lists
    ([], dict(index=[[0, 1]]), get_data_frame_internal_index_multi(index=[(0, 1)])),
    # non empty
    ([], dict(index=[(0, 1), (0, 2)]), get_data_frame_internal_index_multi(index=[(0, 1), (0, 2)])),
    # order matters
    ([], dict(index=[(0, 1), (1, 0)]), get_data_frame_internal_index_multi(index=[(0, 1), (1, 0)])),
])
def test_container_data_indexed_multi_init(args, kwargs, expected):
    container = tahini.container.ContainerDataIndexedMulti(*args, **kwargs)
    assert_frame_equal(container.data_internal, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMulti(), get_data_frame_index_multi()),
    # non empty
    (tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]), get_data_frame_index_multi(index=[(0, 1)])),
    # column
    (
        tahini.container.ContainerDataIndexedMulti(data=dict(a=[1]), index=[(0, 1)]),
        get_data_frame_index_multi(data=dict(a=[1]), index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_data(container, expected):
    assert_frame_equal(container.data, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMulti(), get_data_frame_internal_index_multi()),
    # non empty
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        get_data_frame_internal_index_multi(index=[(0, 1)]),
    ),
    # column
    (
        tahini.container.ContainerDataIndexedMulti(data=dict(a=[1]), index=[(0, 1)]),
        get_data_frame_internal_index_multi(data=dict(a=[1]), index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_data_internal(container, expected):
    assert_frame_equal(container.data_internal, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMulti(), get_data_frame_internal_simple_index_multi()),
    # non empty
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        get_data_frame_internal_simple_index_multi(index=[(0, 1)]),
    ),
    # column
    (
        tahini.container.ContainerDataIndexedMulti(data=dict(a=[1]), index=[(0, 1)]),
        get_data_frame_internal_simple_index_multi(data=dict(a=[1]), index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_data_testing(container, expected):
    assert_frame_equal(container.data_testing, expected)


@pytest.mark.parametrize('container, args, kwargs, type_error, message_error', [
    # index flat
    (
        tahini.container.ContainerDataIndexedMulti(),
        [],
        dict(index=[0]),
        TypeError,
        "object of type 'int' has no len()",
    ),
    # index not exist
    (
        tahini.container.ContainerDataIndexedMulti(),
        [],
        dict(index=[(0, 1)]),
        KeyError,
        "[(0, 1)] not found in axis",
    ),
])
def test_container_data_indexed_multi_drop_error(container, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        container.drop(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMulti(), [], dict(), tahini.container.ContainerDataIndexedMulti()),
    # empty container ignore error
    (
        tahini.container.ContainerDataIndexedMulti(),
        [],
        dict(index=[(0, 1)], errors='ignore'),
        tahini.container.ContainerDataIndexedMulti(),
    ),
    # empty inputs
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        [],
        dict(),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
    ),
    # non empty
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        [],
        dict(index=[(0, 1)]),
        tahini.container.ContainerDataIndexedMulti(),
    ),
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]),
        [],
        dict(index=[(0, 1), (0, 2)]),
        tahini.container.ContainerDataIndexedMulti(),
    ),
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]),
        [],
        dict(index=[(0, 2)]),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
    ),
    # order matters
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        [],
        dict(index=[(1, 0)], errors='ignore'),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
    ),
    # drop columns
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)], columns=['a']),
        [],
        dict(columns=['a']),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_drop(container, args, kwargs, expected):
    container = container.drop(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMulti(), [], dict(), tahini.container.ContainerDataIndexedMulti()),
    # empty inputs
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        [],
        dict(),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
    ),
    # non empty
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        [],
        dict(index=[(0, 2)]),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]),
    ),
    # order matters
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1)]),
        [],
        dict(index=[(1, 0)]),
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (1, 0)]),
    ),
])
def test_container_data_indexed_multi_update(container, args, kwargs, expected):
    container = container.update(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMulti(), [], dict(), tahini.container.ContainerDataIndexedMulti()),
    # non empty
    (
        tahini.container.ContainerDataIndexedMulti(index=[(0, 1), (0, 2)]),
        [],
        dict(mapper={0: 'a', 1: 'b', 2: 'c'}),
        tahini.container.ContainerDataIndexedMulti(index=[('a', 'b'), ('a', 'c')]),
    ),
])
def test_container_data_indexed_multi_map(container, args, kwargs, expected):
    container = container.map(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    # non unique
    ([], dict(index=[(0, 1), (0, 1)]), ValueError, "Index needs to be unique for 'ContainerDataIndexedMultiSets'"),
    # non unique
    ([], dict(index=[(0, 1), (1, 0)]), ValueError, "Index needs to be unique for 'ContainerDataIndexedMultiSets'"),
    # flat index input
    ([], dict(index=[0, 1]), TypeError, "object of type 'int' has no len()"),
    # flat index input
    ([], dict(index=['a', 'b']), ValueError, "Length of names must match number of levels in MultiIndex."),
])
def test_container_data_indexed_multi_sets_init_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.container.ContainerDataIndexedMultiSets(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(), get_data_frame_internal_index_multi_sets()),
    # non empty
    ([], dict(index=[(0, 1)]), get_data_frame_internal_index_multi_sets(index=[(0, 1)])),
    # non empty
    ([], dict(index=[(0, 1), (0, 2)]), get_data_frame_internal_index_multi_sets(index=[(0, 1), (0, 2)])),
])
def test_container_data_indexed_multi_sets_init(args, kwargs, expected):
    container = tahini.container.ContainerDataIndexedMultiSets(*args, **kwargs)
    assert_frame_equal(container.data_internal, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMultiSets(), get_data_frame_index_multi()),
    # non empty
    (tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]), get_data_frame_index_multi(index=[(0, 1)])),
    # column
    (
        tahini.container.ContainerDataIndexedMultiSets(data=dict(a=[1]), index=[(0, 1)]),
        get_data_frame_index_multi(data=dict(a=[1]), index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_sets_data(container, expected):
    assert_frame_equal(container.data, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMultiSets(), get_data_frame_internal_index_multi_sets()),
    # non empty
    (
            tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
            get_data_frame_internal_index_multi_sets(index=[(0, 1)]),
    ),
    # column
    (
            tahini.container.ContainerDataIndexedMultiSets(data=dict(a=[1]), index=[(0, 1)]),
            get_data_frame_internal_index_multi_sets(data=dict(a=[1]), index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_sets_data_internal(container, expected):
    assert_frame_equal(container.data_internal, expected)


@pytest.mark.parametrize('container, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMultiSets(), get_data_frame_internal_simple_index_multi_sets()),
    # non empty
    (
            tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
            get_data_frame_internal_simple_index_multi_sets(index=[(0, 1)]),
    ),
    # column
    (
            tahini.container.ContainerDataIndexedMultiSets(data=dict(a=[1]), index=[(0, 1)]),
            get_data_frame_internal_simple_index_multi_sets(data=dict(a=[1]), index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_sets_data_testing(container, expected):
    assert_frame_equal(container.data_testing, expected)


@pytest.mark.parametrize('container, args, kwargs, type_error, message_error', [
    # index flat
    (
        tahini.container.ContainerDataIndexedMultiSets(),
        [],
        dict(index=[0]),
        TypeError,
        "object of type 'int' has no len()",
    ),
    # index not exist
    (
        tahini.container.ContainerDataIndexedMultiSets(),
        [],
        dict(index=[(0, 1)]),
        KeyError,
        "[frozenset({0, 1})] not found in axis",
    ),
])
def test_container_data_indexed_multi_sets_drop_error(container, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        container.drop(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMultiSets(), [], dict(), tahini.container.ContainerDataIndexedMultiSets()),
    # empty container ignore error
    (
        tahini.container.ContainerDataIndexedMultiSets(),
        [],
        dict(index=[(0, 1)], errors='ignore'),
        tahini.container.ContainerDataIndexedMultiSets(),
    ),
    # empty inputs
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        [],
        dict(),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
    ),
    # non empty
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        [],
        dict(index=[(0, 1)]),
        tahini.container.ContainerDataIndexedMultiSets(),
    ),
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]),
        [],
        dict(index=[(0, 1), (0, 2)]),
        tahini.container.ContainerDataIndexedMultiSets(),
    ),
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]),
        [],
        dict(index=[(0, 2)]),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
    ),
    # order does not matter
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        [],
        dict(index=[(1, 0)]),
        tahini.container.ContainerDataIndexedMultiSets(),
    ),
    # drop columns
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)], columns=['a']),
        [],
        dict(columns=['a']),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_sets_drop(container, args, kwargs, expected):
    container = container.drop(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMultiSets(), [], dict(), tahini.container.ContainerDataIndexedMultiSets()),
    # empty inputs
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        [],
        dict(),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
    ),
    # todo fix this test
    # (
    #     tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
    #     [],
    #     dict(index=[(0, 2)]),
    #     tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]),
    # ),
    # non empty
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        [],
        dict(index=[(0, 2)]),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 2), (0, 1)]),
    ),
    # order does not matter
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
        [],
        dict(index=[(1, 0)]),
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1)]),
    ),
])
def test_container_data_indexed_multi_sets_update(container, args, kwargs, expected):
    container = container.update(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty
    (tahini.container.ContainerDataIndexedMultiSets(), [], dict(), tahini.container.ContainerDataIndexedMultiSets()),
    # non empty
    (
        tahini.container.ContainerDataIndexedMultiSets(index=[(0, 1), (0, 2)]),
        [],
        dict(mapper={0: 'a', 1: 'b', 2: 'c'}),
        tahini.container.ContainerDataIndexedMultiSets(index=[('a', 'b'), ('a', 'c')]),
    ),
])
def test_container_data_indexed_multi_sets_map(container, args, kwargs, expected):
    container = container.map(*args, **kwargs)
    tahini.testing.assert_container_equal(container, expected)

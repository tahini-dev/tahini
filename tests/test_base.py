from math import nan

import pytest
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.pandas import indexes, columns, data_frames
import pandas as pd

import tahini.base


@pytest.mark.parametrize('args, kwargs, type_error, message_error', [
    (
        [],
        dict(index=[], data=pd.DataFrame()),
        ValueError,
        (
            "If input 'data' is 'pandas.DataFrame' then input 'index' has to be 'None' for initializing "
            "'ContainerDataIndexed'"
        ),
    ),
])
def test_collection_data_indexed_init_error(args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        tahini.base.ContainerDataIndexed(*args, **kwargs)
    assert e.value.args[0] == message_error


def get_data_frame(*args, name_index=None, **kwargs) -> pd.DataFrame:
    return pd.DataFrame(*args, **kwargs).rename_axis(index=name_index)


@pytest.mark.parametrize('args, kwargs, expected', [
    # inputs empty
    ([], dict(), get_data_frame()),
    # args - nodes empty
    ([[]], dict(), get_data_frame()),
    # args - data empty
    ([None, []], dict(), get_data_frame()),
    # index empty
    ([], dict(index=[]), get_data_frame()),
    # data empty
    ([], dict(data=[]), get_data_frame()),
    # empty index and data
    ([], dict(index=[], data=[]), get_data_frame()),
    # idempotent with empty
    ([], dict(index=tahini.base.ContainerDataIndexed()), get_data_frame()),
    # idempotent with index
    ([], dict(index=tahini.base.ContainerDataIndexed(index=[0, 1])), get_data_frame(index=[0, 1])),
    # idempotent with index and data dict
    (
        [],
        dict(index=tahini.base.ContainerDataIndexed(index=[0, 1], data=dict(value=['a', 'b']))),
        get_data_frame(data=dict(value=['a', 'b']), index=[0, 1]),
    ),
    # idempotent with data data_frame
    (
        [],
        dict(index=tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']), index=[0, 1]))),
        get_data_frame(data=dict(value=['a', 'b']), index=[0, 1]),
    ),
    # index non unique
    ([], dict(index=[0, 0]), get_data_frame(index=[0, 0])),
    # data dict
    ([], dict(data=dict(value=['a', 'b'])), get_data_frame(dict(value=['a', 'b']))),
    # data records
    ([], dict(data=['a', 'b']), get_data_frame({0: ['a', 'b']})),
    ([], dict(data=[['a'], ['b']]), get_data_frame({0: ['a', 'b']})),
    ([], dict(data=[['a', 'b']]), get_data_frame({0: ['a'], 1: ['b']})),
    # data data_frame
    ([], dict(data=pd.DataFrame(dict(value=['a', 'b']))), get_data_frame(dict(value=['a', 'b']))),
    # data data_frame with index name
    (
        [],
        dict(data=pd.DataFrame(dict(value=['a', 'b'])).rename_axis(index='test')),
        get_data_frame(dict(value=['a', 'b'])),
    ),
    # data dict with index
    ([], dict(index=[1, 2], data=dict(value=['a', 'b'])), get_data_frame(dict(value=['a', 'b']), index=[1, 2])),
    # # data data_frame and index this does not work right now as you get
    # # pd.DataFrame(dict(value=[nan, 'a']), index=[1, 2]) instead of pd.DataFrame(dict(value=['a', 'b']), index=[1, 2])
    # (
    #     [],
    #     dict(index=[1, 2], data=pd.DataFrame(dict(value=['a', 'b']))),
    #     pd.DataFrame(dict(value=['a', 'b']), index=[1, 2]),
    # ),
])
def test_collection_data_indexed_init_simple(args, kwargs, expected):
    container = tahini.base.ContainerDataIndexed(*args, **kwargs)
    pd.testing.assert_frame_equal(container.data, expected)


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
    st.floats,
    st.fractions,
    st.integers,
    st.none,
    st.randoms,
    st.text,
    st.times,
    st.uuids,
)

elements_specific = (
    (
        st.timedeltas,
        dict(min_value=pd.Timedelta.min.to_pytimedelta(), max_value=pd.Timedelta.max.to_pytimedelta()),
        lambda x: True,
    ),
    (
        st.decimals,
        dict(),
        lambda x: not x.is_snan(),
    ),
)


@pytest.mark.parametrize('type_index', types_index)
@pytest.mark.parametrize('elements, kwargs_elements, filter_elements', [
    *((item, dict(), lambda x: True) for item in elements_non_specific),
    *elements_specific,
])
@given(data=st.data())
def test_collection_data_indexed_init_index_single_elements_type(
        type_index,
        elements,
        kwargs_elements,
        filter_elements,
        data,
):
    container = tahini.base.ContainerDataIndexed(
        index=data.draw(type_index(elements=elements(**kwargs_elements).filter(filter_elements))),
    )
    assert isinstance(container.data, pd.DataFrame)


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
])
@given(data=st.data())
def test_collection_data_indexed_init_index_single_elements_type_xfail(type_index, elements, data):
    container = tahini.base.ContainerDataIndexed(index=data.draw(type_index(elements=elements())))
    assert isinstance(container.data, pd.DataFrame)


@pytest.fixture(scope='module')
def list_elements():
    output_value = (
        *(elements() for elements in elements_non_specific),
        *(item[0](**item[1]).filter(item[2]) for item in elements_specific),
    )
    return output_value


@pytest.mark.parametrize('type_index', types_index)
@given(data=st.data())
def test_nodes_init_index_multiple_elements_type(type_index, list_elements, data):
    container = tahini.base.ContainerDataIndexed(
        index=data.draw(type_index(elements=st.one_of(*list_elements)))
    )
    assert isinstance(container.data, pd.DataFrame)


@given(data=data_frames(columns=(columns('A', elements=st.integers())), index=indexes(elements=st.integers())))
def test_nodes_init_data_data_frame(data):
    container = tahini.base.ContainerDataIndexed(data=data)
    assert isinstance(container.data, pd.DataFrame)


def container_data_indexed_empty():
    return tahini.base.ContainerDataIndexed()


def container_data_indexed_range(size=10):
    return tahini.base.ContainerDataIndexed(index=range(size))


class ChildContainerDataIndexed(tahini.base.ContainerDataIndexed):
    @staticmethod
    def _name_index():
        return 'child'


@pytest.mark.parametrize('container, expected', [
    # empty
    (container_data_indexed_empty(), f'ContainerDataIndexed(index={pd.Index([])})'),
    # non empty
    (container_data_indexed_range(), f'ContainerDataIndexed(index={pd.RangeIndex(stop=10)})'),
    # child class
    (ChildContainerDataIndexed(), f'ChildContainerDataIndexed(index={pd.Index([], name="child")})'),
])
def test_container_data_indexed_repr(container, expected):
    actual = repr(container)
    assert actual == expected


@pytest.mark.parametrize('container, expected', [
    # empty
    (container_data_indexed_empty(), []),
    # non empty
    (container_data_indexed_range(), list(range(10))),
])
def test_container_data_indexed_iter(container, expected):
    assert [item for item in container] == expected


@pytest.mark.parametrize('container, item, expected', [
    # not in empty
    (container_data_indexed_empty(), 0, False),
    # in range
    (container_data_indexed_range(), 0, True),
    # not in range
    (container_data_indexed_range(), 11, False),
])
def test_container_data_indexed_contains(container, item, expected):
    assert (item in container) == expected


@pytest.mark.parametrize('container, expected', [
    # empty
    (container_data_indexed_empty(), 0),
    # non empty
    (container_data_indexed_range(), 10),
])
def test_container_data_indexed_len(container, expected):
    assert len(container) == expected


@pytest.mark.parametrize('container_left, container_right, expected', [
    # empty
    (container_data_indexed_empty(), container_data_indexed_empty(), True),
    # non empty
    (container_data_indexed_range(), container_data_indexed_range(), True),
    # empty versus non empty
    (container_data_indexed_empty(), container_data_indexed_range(), False),
    # None right
    (container_data_indexed_range(), None, False),
    # None left
    (None, container_data_indexed_range(), False),
    # different order
    (tahini.base.ContainerDataIndexed(index=[1, 2]), tahini.base.ContainerDataIndexed(index=[2, 1]), True),
    # child class - empty
    (ChildContainerDataIndexed(), ChildContainerDataIndexed(), True),
    # child class - None
    (ChildContainerDataIndexed(), None, False),
])
def test_container_data_indexed_eq(container_left, container_right, expected):
    assert (container_left == container_right) == expected


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # idempotent on empty set
    (container_data_indexed_empty(), [], dict(), get_data_frame()),
    # simple update
    (container_data_indexed_empty(), [], dict(index=[0, 1]), get_data_frame(index=[0, 1])),
    # idempotent update
    (container_data_indexed_range(2), [], dict(index=container_data_indexed_range(2)), get_data_frame(index=range(2))),
    # update seems to sort index
    (container_data_indexed_range(2), [], dict(index=[4, 3, 2]), get_data_frame(index=[0, 1, 2, 3, 4])),
    # non unique index left
    (
        tahini.base.ContainerDataIndexed(index=[0, 0, 1]),
        [],
        dict(index=[3, 2, 0]),
        get_data_frame(index=[0, 0, 1, 2, 3]),
    ),
    # non unique index right
    (
        tahini.base.ContainerDataIndexed(index=[0, 1]),
        [], dict(index=[3, 2, 0, 0]),
        get_data_frame(index=[0, 0, 1, 2, 3]),
    ),
    # non unique index both
    (
        tahini.base.ContainerDataIndexed(index=[0, 0, 1]),
        [],
        dict(index=[3, 2, 0, 0]),
        get_data_frame(index=[0, 0, 0, 0, 1, 2, 3]),
    ),
    # new column for empty index
    (
        container_data_indexed_empty(),
        [],
        dict(data=pd.DataFrame(dict(value=['a', 'b']))),
        get_data_frame(dict(value=['a', 'b'])),
    ),
    # New index item and column for existing index
    (
        container_data_indexed_range(2),
        [],
        dict(data=pd.DataFrame(dict(value=['a', 'b']))),
        get_data_frame(dict(value=['a', 'b']))
    ),
    # new index item and column
    (
        container_data_indexed_range(2),
        [],
        dict(data=pd.DataFrame(dict(value=['a', 'b']), index=[1, 2])),
        get_data_frame(dict(value=[nan, 'a', 'b'])),
    ),
    # cannot update to nan with default func
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value=[nan]))),
        get_data_frame(dict(value=['a', 'b'])),
    ),
    # single value in column
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value=['c']))),
        get_data_frame(dict(value=['c', 'b'])),
    ),
    # single value in column
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value=['c']), index=[1])),
        get_data_frame(dict(value=['a', 'c'])),
    ),
    # new column
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value_2=['c', 'd']))),
        get_data_frame(dict(value=['a', 'b'], value_2=['c', 'd'])),
    ),
    # row update
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b'], value_2=['c', 'd']))),
        [],
        dict(data=pd.DataFrame(dict(value=['e'], value_2=['f']))),
        get_data_frame(dict(value=['e', 'b'], value_2=['f', 'd'])),
    ),
    # child - empty
    (ChildContainerDataIndexed(), [], dict(), get_data_frame(name_index='child')),
    # child - non empty
    (ChildContainerDataIndexed(index=[0, 1]), [], dict(index=[2]), get_data_frame(index=[0, 1, 2], name_index='child')),
])
def test_container_data_indexed_update(container, args, kwargs, expected):
    container_updated = container.update(*args, **kwargs)
    assert isinstance(container_updated, container.__class__)
    pd.testing.assert_frame_equal(container_updated.data, expected)


@pytest.mark.parametrize('container, args, kwargs, type_error, message_error', [
    # cannot pass empty arguments
    (
        container_data_indexed_empty(),
        [],
        dict(),
        ValueError,
        "Need to specify at least one of 'labels', 'index' or 'columns'",
    ),
    # raises error if the node is not found, this can be suppressed by using errors='ignore'
    (container_data_indexed_empty(), [], dict(index=[1]), KeyError, "[1] not found in axis"),
])
def test_container_data_indexed_drop_error(container, args, kwargs, type_error, message_error):
    with pytest.raises(type_error) as e:
        container.drop(*args, **kwargs)
    assert e.value.args[0] == message_error


@pytest.mark.parametrize('container, args, kwargs, expected', [
    # empty list
    (container_data_indexed_empty(), [], dict(index=[]), get_data_frame()),
    # basic example
    (container_data_indexed_range(1), [], dict(index=[0]), get_data_frame(index=pd.Int64Index([]))),
    # another basic example
    (container_data_indexed_range(2), [], dict(index=[0]), get_data_frame(index=[1])),
    # ignore error if node not found
    (container_data_indexed_empty(), [], dict(index=[1], errors='ignore'), get_data_frame()),
    # node not in container
    (container_data_indexed_range(1), [], dict(index=0), get_data_frame(index=pd.Int64Index([]))),
    # multiple container_data_indexed
    (container_data_indexed_range(3), [], dict(index=[1, 2]), get_data_frame(index=[0])),
    # column
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(columns=['value']),
        get_data_frame(index=range(2)),
    ),
    # column not in container
    (
        tahini.base.ContainerDataIndexed(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(columns='value'),
        get_data_frame(index=range(2)),
    ),
    # child
    (ChildContainerDataIndexed(index=[0, 1]), [], dict(index=[0]), get_data_frame(index=[1], name_index='child'))
])
def test_container_data_indexed_drop(container, args, kwargs, expected):
    container_after_dropped = container.drop(*args, **kwargs)
    assert isinstance(container_after_dropped, container.__class__)
    pd.testing.assert_frame_equal(container_after_dropped.data, expected)


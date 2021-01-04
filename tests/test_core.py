import pytest
from hypothesis import given
import hypothesis.strategies as st
from pandas import Timedelta

import tahini.core


@pytest.mark.parametrize('args, kwargs, error_type, error_message', [
    ([], dict(), ValueError, 'Need to provide at least "data" or "index" or "size" for initializing "Nodes".'),
    ([], dict(size=0.5), TypeError, "Wrong type <class 'float'> for value 0.5"),
])
def test_nodes_init(args, kwargs, error_type, error_message):

    with pytest.raises(error_type) as e:
        tahini.core.Nodes(*args, **kwargs)

    assert e.value.args[0] == error_message


@pytest.mark.parametrize('args, kwargs', [
    ([[]], dict()),
    ([None, []], dict()),
    ([None, None, 1], dict()),
    ([], dict(data=[])),
    ([], dict(index=[])),
    ([], dict(data=[], index=[])),
    ([], dict(data=[], size=1)),
    ([], dict(data=[], size=-1)),
    ([], dict(data=[], size=0)),
])
def test_nodes_init_simple(args, kwargs):
    nodes = tahini.core.Nodes(*args, **kwargs)
    assert isinstance(nodes, tahini.core.Nodes)


containers_hashable = [
    st.frozensets,
    st.sets,
]

containers_non_hashable = [
    st.iterables,
    st.lists,
    st.tuples,
]

containers_all = containers_hashable + containers_non_hashable


@pytest.mark.parametrize('container_type', containers_non_hashable)
@pytest.mark.parametrize('elements, elements_kwargs', [
    (st.binary, dict()),
    (st.booleans, dict()),
    (st.characters, dict()),
    (st.complex_numbers, dict()),
    (st.dates, dict()),
    (st.datetimes, dict()),
    (st.decimals, dict()),
    (st.floats, dict()),
    (st.fractions, dict()),
    (st.integers, dict()),
    (st.text, dict()),
    (st.timedeltas, dict(min_value=Timedelta.min.to_pytimedelta(), max_value=Timedelta.max.to_pytimedelta())),
    (st.times, dict()),
])
@given(data=st.data())
def test_nodes_init_index_multiple_container_non_hashable_and_data_type(
        container_type,
        elements,
        elements_kwargs,
        data,
):
    nodes = tahini.core.Nodes(index=data.draw(container_type(elements(**elements_kwargs))))
    assert isinstance(nodes, tahini.core.Nodes)


@pytest.mark.parametrize('container_type', containers_hashable)
@pytest.mark.parametrize('elements, elements_kwargs', [
    (st.binary, dict()),
    (st.booleans, dict()),
    (st.characters, dict()),
    (st.complex_numbers, dict()),
    (st.dates, dict()),
    (st.datetimes, dict()),
    (st.decimals, dict(allow_nan=False)),
    (st.floats, dict()),
    (st.fractions, dict()),
    (st.integers, dict()),
    (st.text, dict()),
    (st.timedeltas, dict(min_value=Timedelta.min.to_pytimedelta(), max_value=Timedelta.max.to_pytimedelta())),
    (st.times, dict()),
])
@given(data=st.data())
def test_nodes_init_index_multiple_container_hashable_and_data_type(
        container_type,
        elements,
        elements_kwargs,
        data,
):
    nodes = tahini.core.Nodes(index=data.draw(container_type(elements(**elements_kwargs))))
    assert isinstance(nodes, tahini.core.Nodes)


@pytest.mark.xfail
@pytest.mark.parametrize('container_type, elements', [
    # pandas.Timedeltas max and min do not match python standard library datetime.timedelta max and min
    *[(container, st.timedeltas) for container in containers_all],
    # error with decimals with sets and frozensets
    *[
        (st.sets, st.decimals),
        (st.frozensets, st.decimals),
    ],
])
@given(data=st.data())
def test_nodes_init_index_multiple_xfail(container_type, elements, data):
    nodes = tahini.core.Nodes(index=data.draw(container_type(elements())))
    assert isinstance(nodes, tahini.core.Nodes)

from decimal import Decimal

import pytest
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.pandas import indexes
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


containers_hashable = (
    st.frozensets,
    st.sets,
    indexes,
)

containers_non_hashable = (
    st.iterables,
    st.lists,
)

containers_all = containers_hashable + containers_non_hashable

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
    (st.timedeltas, dict(min_value=Timedelta.min.to_pytimedelta(), max_value=Timedelta.max.to_pytimedelta())),
)


@pytest.mark.parametrize('container_type', containers_all)
@pytest.mark.parametrize('elements, kwargs_elements', [
    *((item, dict()) for item in elements_non_specific),
    *elements_specific,
])
@given(data=st.data())
def test_nodes_init_index_single_elements_type(
        container_type,
        elements,
        kwargs_elements,
        data,
):
    nodes = tahini.core.Nodes(index=data.draw(container_type(elements=elements(**kwargs_elements))))
    assert isinstance(nodes, tahini.core.Nodes)


@pytest.mark.parametrize('container_type, elements, kwargs_elements, filter_elements', [
    *((container, st.decimals, dict(), lambda x: True) for container in containers_non_hashable),
    *((container, st.decimals, dict(), lambda x: not Decimal.is_snan(x)) for container in containers_hashable),
])
@given(data=st.data())
def test_nodes_init_index_single_elements_type_specific(
        container_type,
        elements,
        kwargs_elements,
        filter_elements,
        data,
):
    nodes = tahini.core.Nodes(
        index=data.draw(container_type(elements=elements(**kwargs_elements).filter(filter_elements)))
    )
    assert isinstance(nodes, tahini.core.Nodes)


@pytest.mark.xfail
@pytest.mark.parametrize('container_type, elements', [
    # pandas.Timedeltas max and min do not match python standard library datetime.timedelta max and min
    *((container, st.timedeltas) for container in containers_all),
    # error with decimals with sets and frozensets
    *(
        (st.sets, st.decimals),
        (st.frozensets, st.decimals),
    ),
])
@given(data=st.data())
def test_nodes_init_index_single_elements_type_xfail(container_type, elements, data):
    nodes = tahini.core.Nodes(index=data.draw(container_type(elements=elements())))
    assert isinstance(nodes, tahini.core.Nodes)


@pytest.fixture(scope='module')
def list_elements():
    output_value = (
        *(elements() for elements in elements_non_specific),
        *(item[0](**item[1]) for item in elements_specific),
    )
    return output_value


@pytest.mark.parametrize('container_type, list_elements_specific', [
    *((container, (st.decimals(),)) for container in containers_non_hashable),
    *((container, (st.decimals().filter(lambda x: not Decimal.is_snan(x)),)) for container in containers_non_hashable),
])
@given(data=st.data())
def test_nodes_init_index_multiple_elements_type(container_type, list_elements_specific, list_elements, data):
    nodes = tahini.core.Nodes(
        index=data.draw(container_type(elements=st.one_of(*list_elements, *list_elements_specific)))
    )
    assert isinstance(nodes, tahini.core.Nodes)

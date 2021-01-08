from decimal import Decimal
from math import nan

import pytest
from hypothesis import given
import hypothesis.strategies as st
from hypothesis.extra.pandas import indexes, columns, data_frames
import pandas as pd

import tahini.core


@pytest.mark.parametrize('args, kwargs, error_type, error_message', [
    # can only pass empty or nodes and/or data or size
    (
        [],
        dict(nodes=[], data=[], size=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'nodes', 'data', 'nodes' and 'data' or 'size'",
    ),
    # can only pass empty or nodes and/or data or size
    (
        [],
        dict(nodes=[], size=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'nodes', 'data', 'nodes' and 'data' or 'size'",
    ),
    # can only pass empty or nodes and/or data or size
    (
        [],
        dict(data=[], size=1),
        ValueError,
        "Inputs for 'Nodes' can either be empty or contain 'nodes', 'data', 'nodes' and 'data' or 'size'",
    ),
    # size errors are driven by range(stop=size)
    ([], dict(size=0.5), TypeError, "Wrong type <class 'float'> for value 0.5"),
    # if data is a data_frame then nodes has to be None because no simple way to set index to nodes without taking into
    # account other cases
    (
        [],
        dict(data=pd.DataFrame(), nodes=[]),
        ValueError,
        "'nodes' has to be None if 'data' is a DataFrame for 'Nodes'",
    ),
])
def test_nodes_init(args, kwargs, error_type, error_message):
    with pytest.raises(error_type) as e:
        tahini.core.Nodes(*args, **kwargs)
    assert e.value.args[0] == error_message


@pytest.mark.parametrize('args, kwargs, expected', [
    # inputs empty
    ([], dict(), pd.DataFrame()),
    # args - nodes empty
    ([[]], dict(), pd.DataFrame()),
    # args - data empty
    ([None, []], dict(), pd.DataFrame()),
    # args - size
    ([None, None, 1], dict(), pd.DataFrame(index=range(1))),
    # nodes empty
    ([], dict(nodes=[]), pd.DataFrame()),
    # data empty
    ([], dict(data=[]), pd.DataFrame()),
    # empty nodes and data
    ([], dict(nodes=[], data=[]), pd.DataFrame()),
    # size zero
    ([], dict(size=0), pd.DataFrame(index=range(0))),
    # size
    ([], dict(size=1), pd.DataFrame(index=range(1))),
    # size negative
    ([], dict(size=-1), pd.DataFrame(index=range(-1))),
    # nodes input it's own class
    ([], dict(nodes=tahini.core.Nodes()), pd.DataFrame()),
    # nodes non unique
    ([], dict(nodes=[0, 0]), pd.DataFrame(index=[0, 0])),
    # data data_frame
    ([], dict(data=pd.DataFrame(dict(value=['a', 'b']))), pd.DataFrame(dict(value=['a', 'b']))),
    # data dict
    ([], dict(data=dict(value=['a', 'b'])), pd.DataFrame(dict(value=['a', 'b']))),
    # data dict with nodes
    ([], dict(nodes=[1, 2], data=dict(value=['a', 'b'])), pd.DataFrame(dict(value=['a', 'b']), index=[1, 2])),
])
def test_nodes_init_simple(args, kwargs, expected):
    nodes = tahini.core.Nodes(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes.data, expected)


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
    (st.timedeltas, dict(min_value=pd.Timedelta.min.to_pytimedelta(), max_value=pd.Timedelta.max.to_pytimedelta())),
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
    nodes = tahini.core.Nodes(nodes=data.draw(container_type(elements=elements(**kwargs_elements))))
    assert isinstance(nodes.data, pd.DataFrame)


@pytest.mark.parametrize('container_type, elements, kwargs_elements, filter_elements', [
    *((container, st.decimals, dict(), lambda x: not Decimal.is_snan(x)) for container in containers_all),
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
        nodes=data.draw(container_type(elements=elements(**kwargs_elements).filter(filter_elements)))
    )
    assert isinstance(nodes.data, pd.DataFrame)


@pytest.mark.xfail
@pytest.mark.parametrize('container_type, elements', [
    # pandas.Timedeltas max and min do not match python standard library datetime.timedelta max and min
    *((container, st.timedeltas) for container in containers_all),
    # error with decimals not being able to hash snan
    *((container, st.decimals) for container in containers_all),
])
@given(data=st.data())
def test_nodes_init_index_single_elements_type_xfail(container_type, elements, data):
    nodes = tahini.core.Nodes(nodes=data.draw(container_type(elements=elements())))
    assert isinstance(nodes.data, pd.DataFrame)


@pytest.fixture(scope='module')
def list_elements():
    output_value = (
        *(elements() for elements in elements_non_specific),
        *(item[0](**item[1]) for item in elements_specific),
    )
    return output_value


@pytest.mark.parametrize('container_type, list_elements_specific', [
    *((container, (st.decimals().filter(lambda x: not Decimal.is_snan(x)),)) for container in containers_all),
])
@given(data=st.data())
def test_nodes_init_index_multiple_elements_type(container_type, list_elements_specific, list_elements, data):
    nodes = tahini.core.Nodes(
        nodes=data.draw(container_type(elements=st.one_of(*list_elements, *list_elements_specific)))
    )
    assert isinstance(nodes.data, pd.DataFrame)


@given(data=data_frames(columns=(columns('A', elements=st.integers())), index=indexes(elements=st.integers())))
def test_nodes_init_data(data):
    nodes = tahini.core.Nodes(data=data)
    assert isinstance(nodes.data, pd.DataFrame)


def nodes_empty():
    return tahini.core.Nodes(nodes=[])


def nodes_range(size=10):
    return tahini.core.Nodes(size=size)


@pytest.mark.parametrize('nodes, expected', [
    (nodes_empty(), f'Nodes(index={pd.Index([])})'),
    (nodes_range(10), f'Nodes(index={pd.RangeIndex(stop=10)})'),
])
def test_nodes_repr(nodes, expected):
    actual = repr(nodes)
    assert actual == expected


@pytest.mark.parametrize('nodes, expected', [
    (nodes_empty(), []),
    (nodes_range(10), list(range(10))),
])
def test_nodes_iter(nodes, expected):
    assert [item for item in nodes] == expected


@pytest.mark.parametrize('nodes, item, expected', [
    (nodes_empty(), 0, False),
    (nodes_range(3), 0, True),
    (nodes_range(3), 3, False),
])
def test_nodes_contains(nodes, item, expected):
    assert (item in nodes) == expected


@pytest.mark.parametrize('nodes, expected', [
    (nodes_empty(), 0),
    (nodes_range(3), 3),
])
def test_nodes_len(nodes, expected):
    assert len(nodes) == expected


@pytest.mark.parametrize('nodes_1, nodes_2, expected', [
    (nodes_empty(), nodes_empty(), True),
    (nodes_range(3), nodes_range(3), True),
    (nodes_empty(), nodes_range(3), False),
    (nodes_range(3), None, False),
    (None, nodes_range(3), False),
    (None, nodes_range(3), False),
    (tahini.core.Nodes(nodes=[1, 2]), tahini.core.Nodes(nodes=[2, 1]), True),
])
def test_nodes_eq(nodes_1, nodes_2, expected):
    assert (nodes_1 == nodes_2) == expected


@pytest.mark.parametrize('nodes, args, kwargs, expected', [
    # idempotent on empty set
    (nodes_empty(), [], dict(), pd.DataFrame()),
    # simple update
    (nodes_empty(), [], dict(nodes=[0, 1]), pd.DataFrame(index=[0, 1])),
    # idempotent update
    (nodes_range(2), [], dict(nodes=nodes_range(2)), pd.DataFrame(index=range(2))),
    # update seems to sort index
    (nodes_range(2), [], dict(nodes=[4, 3, 2]), pd.DataFrame(index=[0, 1, 2, 3, 4])),
    # non unique nodes left
    (tahini.core.Nodes(nodes=[0, 0, 1]), [], dict(nodes=[3, 2, 0]), pd.DataFrame(index=[0, 0, 1, 2, 3])),
    # non unique nodes right
    (tahini.core.Nodes(nodes=[0, 1]), [], dict(nodes=[3, 2, 0, 0]), pd.DataFrame(index=[0, 0, 1, 2, 3])),
    # non unique nodes both
    (tahini.core.Nodes(nodes=[0, 0, 1]), [], dict(nodes=[3, 2, 0, 0]), pd.DataFrame(index=[0, 0, 0, 0, 1, 2, 3])),
    # new column
    (nodes_empty(), [], dict(data=pd.DataFrame(dict(value=['a', 'b']))), pd.DataFrame(dict(value=['a', 'b']))),
    # new node and column
    (nodes_range(2), [], dict(data=pd.DataFrame(dict(value=['a', 'b']))), pd.DataFrame(dict(value=['a', 'b']))),
    (
        nodes_range(2),
        [],
        dict(data=pd.DataFrame(dict(value=['a', 'b']), index=[1, 2])),
        pd.DataFrame(dict(value=[nan, 'a', 'b'])),
    ),
    # cannot update to nan with default func
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value=[nan]))),
        pd.DataFrame(dict(value=['a', 'b'])),
    ),
    # single value in column
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value=['c']))),
        pd.DataFrame(dict(value=['c', 'b'])),
    ),
    # single value in column
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value=['c']), index=[1])),
        pd.DataFrame(dict(value=['a', 'c'])),
    ),
    # new column
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(data=pd.DataFrame(dict(value_2=['c', 'd']))),
        pd.DataFrame(dict(value=['a', 'b'], value_2=['c', 'd'])),
    ),
    # row update
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b'], value_2=['c', 'd']))),
        [],
        dict(data=pd.DataFrame(dict(value=['e'], value_2=['f']))),
        pd.DataFrame(dict(value=['e', 'b'], value_2=['f', 'd'])),
    ),
])
def test_nodes_update(nodes, args, kwargs, expected):
    nodes_updated = nodes.update(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes_updated.data, expected)


@pytest.mark.parametrize('nodes, args, kwargs, error_type, error_message', [
    # cannot pass empty arguments
    (nodes_empty(), [], dict(), ValueError, "Need to specify at least one of 'labels', 'index' or 'columns'"),
    # raises error if the node is not found, this can be suppressed by using errors='ignore'
    (nodes_empty(), [], dict(nodes=[1]), KeyError, "[1] not found in axis"),
])
def test_nodes_drop_error(nodes, args, kwargs, error_type, error_message):
    with pytest.raises(error_type) as e:
        nodes.drop(*args, **kwargs)
    assert e.value.args[0] == error_message


@pytest.mark.parametrize('nodes, args, kwargs, expected', [
    # empty list
    (nodes_empty(), [], dict(nodes=[]), pd.DataFrame()),
    # basic example
    (nodes_range(1), [], dict(nodes=[0]), pd.DataFrame(index=pd.Int64Index([]))),
    # another basic example
    (nodes_range(2), [], dict(nodes=[0]), pd.DataFrame(index=[1])),
    # ignore error if node not found
    (nodes_empty(), [], dict(nodes=[1], errors='ignore'), pd.DataFrame()),
    # node not in container
    (nodes_range(1), [], dict(nodes=0), pd.DataFrame(index=pd.Int64Index([]))),
    # multiple nodes
    (nodes_range(3), [], dict(nodes=[1, 2]), pd.DataFrame(index=[0])),
    # column
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(columns=['value']),
        pd.DataFrame(index=range(2)),
    ),
    # column not in container
    (
        tahini.core.Nodes(data=pd.DataFrame(dict(value=['a', 'b']))),
        [],
        dict(columns='value'),
        pd.DataFrame(index=range(2)),
    ),
])
def test_nodes_drop(nodes, args, kwargs, expected):
    nodes_after_dropped = nodes.drop(*args, **kwargs)
    pd.testing.assert_frame_equal(nodes_after_dropped.data, expected)


# @pytest.mark.parametrize('args, kwargs', [
#     ([], dict()),
# ])
# def test_graph_init(args, kwargs):
#     graph = tahini.core.Graph(*args, **kwargs)
#     assert isinstance(graph, tahini.core.Graph)

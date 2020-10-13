import pytest
from uuid import UUID

import tahini.core


def test_node_init():
    node = tahini.core.Node()
    assert isinstance(node, tahini.core.Node)


@pytest.mark.parametrize('args, kwargs, error_type, error_message', [
    ([], dict(name={}), TypeError, 'Input "name" needs to be hashable.')
])
def test_node_init_error(args, kwargs, error_type, error_message):
    with pytest.raises(error_type) as e:
        tahini.core.Node(*args, **kwargs)
    assert e.value.args[0] == error_message


@pytest.fixture
def node(args, kwargs):
    return tahini.core.Node(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, attribute, expected', [
    ([], dict(), 'name', UUID),
    ([0], dict(), 'name', int),
    ([], dict(name=0), 'name', int),
    ([], dict(name='a'), 'name', str),
])
def test_node_attribute_type(node, args, kwargs, attribute, expected):
    assert isinstance(getattr(node, attribute), expected)


@pytest.mark.parametrize('args, kwargs, attribute, expected', [
    ([0], dict(), 'name', 0),
    ([], dict(name=0), 'name', 0),
    ([], dict(name='a'), 'name', 'a'),
])
def test_node_attribute(node, args, kwargs, attribute, expected):
    assert getattr(node, attribute) == expected


@pytest.mark.parametrize('args, kwargs, expected', [
    ([], dict(name=0), 'Node(name=0)'),
    ([], dict(name='0'), "Node(name='0')"),
])
def test_node_repr(node, args, kwargs, expected):
    actual = repr(node)
    assert actual == expected


@pytest.fixture
def node_other(args_other, kwargs_other):
    return tahini.core.Node(*args_other, **kwargs_other)


@pytest.mark.parametrize('args, kwargs, args_other, kwargs_other, expected', [
    ([], dict(name=0), [], dict(name=0), True),
    ([], dict(name='0'), [], dict(name=0), False),
    ([], dict(), [], dict(), False),
])
def test_node_repr(node, args, kwargs, node_other, args_other, kwargs_other, expected):
    actual = node == node_other
    assert actual == expected


@pytest.mark.parametrize('args, kwargs, expected', [
    ([], dict(name=0), hash(0)),
])
def test_node_hash(node, args, kwargs, expected):
    actual = hash(node)
    assert actual == expected

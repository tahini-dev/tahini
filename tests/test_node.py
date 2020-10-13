import pytest

import tahini.node


def test_node_init():
    node = tahini.node.Node()
    assert isinstance(node, tahini.node.Node)


@pytest.mark.parametrize('node, attribute', [
    (tahini.node.Node(), '_id'),
    (tahini.node.Node(), 'id'),
])
def test_node_properties(node, attribute):
    assert hasattr(node, attribute)

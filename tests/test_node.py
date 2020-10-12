import tahini.node


def test_node():
    node = tahini.node.Node()
    assert hasattr(node, '_id')

import pytest

import tahini.core.nodes
import tahini.testing


@pytest.mark.parametrize('args, kwargs', [
    # empty
    ([], dict()),
    # order
    ([], dict(order=1)),
])
def test_nodes_init_simple(args, kwargs):
    tahini.core.nodes.Nodes(*args, **kwargs)


@pytest.mark.parametrize('args, kwargs, expected', [
    # empty
    ([], dict(), tahini.core.nodes.Nodes()),
    # order
    ([], dict(order=1), tahini.core.nodes.Nodes(index=[0])),
    ([], dict(order=2), tahini.core.nodes.Nodes(index=[0, 1])),
])
def test_nodes_init(args, kwargs, expected):
    nodes = tahini.core.nodes.Nodes(*args, **kwargs)
    tahini.testing.testing.assert_container_equal(nodes, expected)

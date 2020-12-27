from __future__ import annotations

from abc import ABC
from typing import Optional
from collections.abc import Hashable, Sequence, Mapping
from uuid import uuid4


class Node:

    def __init__(
            self,
            name: Optional[Hashable] = None,
            node: Optional[Node] = None,
    ):
        if node is not None:
            name = node.name
        if name is None:
            name = uuid4()
        if not isinstance(name, Hashable):
            raise TypeError('Input "name" needs to be hashable.')
        self._name = name

    @property
    def name(self) -> Hashable:
        return self._name

    def __repr__(self):
        return f'{self.__class__.__name__}(name={self.name!r})'

    def __eq__(self, other: Node):
        return self.name == other.name

    def __hash__(self):
        if getattr(self, '_hash', None) is None:
            self._hash = hash(self.name)
        return self._hash


class Nodes(Mapping, ABC):

    def __init__(
            self,
            nodes: Optional[Sequence[Node]] = None,
    ):
        if nodes is None:
            nodes = dict()
        elif isinstance(nodes, Sequence):
            nodes = {node.name: node for node in nodes}
        else:
            raise NotImplemented

        self._nodes = nodes

    def __len__(self):
        return len(self._nodes)

    def __iter__(self):
        return iter(self._nodes)

    def __getitem__(self, key):
        return self._nodes[key]

    def __repr__(self):
        return f'{self.__class__.__name__}[{self.__len__()}]{self._nodes!r}'

    def __hash__(self):
        # It would have been simpler and maybe more obvious to
        # use hash(tuple(sorted(self._d.iteritems()))) from this discussion
        # so far, but this solution is O(n). I don't know what kind of
        # n we are going to run into, but sometimes it's hard to resist the
        # urge to optimize when it will gain improved algorithmic performance.
        if getattr(self, '_hash', None) is None:
            hash_ = 0
            for pair in self.items():
                hash_ ^= hash(pair)
            self._hash = hash_
        return self._hash

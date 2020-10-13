from __future__ import annotations
from typing import Optional
from collections.abc import Hashable, Sequence
from uuid import uuid4


class Node:

    def __init__(
            self,
            name: Optional[Hashable] = None,
    ):

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
        return hash(self.name)

from __future__ import annotations
from uuid import uuid4, UUID


class Node:
    def __init__(self):
        self._id = uuid4()

    @property
    def id(self) -> UUID:
        return self._id

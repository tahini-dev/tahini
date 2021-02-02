from typing import Optional, Sequence, Tuple

import pandas as pd
import numpy as np

from .core import Graph

__all__ = ['get_positions']


# All credit for this goes to networkx https://github.com/networkx/networkx/blob/master/networkx/drawing/layout.py


def _process_parameters(
        center: Optional[Sequence] = None,
        dim: Optional[int] = None,
) -> Tuple[np.array, int]:

    if dim is None:
        dim = 2

    if center is None:
        center = np.zeros(dim)
    else:
        center = np.asarray(center)

    if center.shape != (dim,):
        raise ValueError("Shape of 'center' has to be ('dim',)")

    return center, dim


def _array_to_data_frame(
        array: np.array,
        graph: Graph,
        dim: int,
) -> pd.DataFrame:

    df = pd.DataFrame(
        data=array,
        index=graph.nodes.data.index,
        columns=[f'position_dim_{i}' for i in range(dim)],
    )

    return df


def get_positions(
        graph: Graph,
        layout: Optional[str] = None,
        center: Optional[Sequence] = None,
        dim: Optional[int] = None,
        **kwargs,
) -> pd.DataFrame:

    if layout is None:
        layout = 'circular'

    center, dim = _process_parameters(center=center, dim=dim)

    function_get_positions_array = globals()[f'_get_positions_layout_{layout}']
    positions_base = function_get_positions_array(graph=graph, dim=dim, **kwargs)

    positions = positions_base + center

    df = _array_to_data_frame(array=positions, graph=graph, dim=dim)

    return df


def _get_positions_layout_circular(
        graph: Graph,
        dim: int,
        scale: Optional[int] = None,
) -> np.array:

    if scale is None:
        scale = 1

    if dim < 2:
        raise ValueError("'dim' has to be > 1 for circular layout for graph positions")

    pad_dims = max(0, dim - 2)

    if graph.order == 1:
        positions = np.zeros((1, dim))
    else:
        theta = np.linspace(start=0, stop=1, num=graph.order, endpoint=False) * 2 * np.pi
        positions = np.column_stack(
            [np.cos(theta), np.sin(theta), np.zeros((graph.order, pad_dims))]
        )

    positions = positions * scale

    return positions


def _get_positions_layout_random(
        graph: Graph,
        dim: int,
        seed: Optional[int] = None,
) -> np.array:
    rs = np.random.RandomState(seed=seed)
    positions = rs.rand(graph.order, dim)
    return positions

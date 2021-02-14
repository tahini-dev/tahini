from __future__ import annotations
from typing import TYPE_CHECKING, Optional

import pandas as pd

if TYPE_CHECKING:
    from ..core.graph import TypeGraph


class Base:

    def __init__(
            self,
            graph: TypeGraph,
    ):
        self.graph = graph

    def get_data_frame_plot(
            self,
            **kwargs,
    ) -> pd.DataFrame:
        df = self.graph.nodes.update(data=self.graph.nodes.get_positions()).data
        return df

    def plot(self, **kwargs):
        ...


class Plotly(Base):

    def plot(
            self,
            df: Optional[pd.DataFrame],
            x: Optional[str],
            y: Optional[str],
            **kwargs,
    ):
        import plotly.express as px

        if df is None:
            df = self.get_data_frame_plot()

        if x is None:
            x = 'position_dim_0'

        if y is None:
            y = 'position_dim_1'

        fig = px.scatter(
            df,
            x=x,
            y=y,
            **kwargs,
        )

        return fig

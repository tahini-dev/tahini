from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict

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
            *args,
            **kwargs,
    ) -> pd.DataFrame:
        df = self.graph.nodes.update(data=self.graph.nodes.get_positions()).data
        return df

    def plot(self, **kwargs):
        ...


class Plotly(Base):

    def plot(
            self,
            df: Optional[pd.DataFrame] = None,
            x: Optional[str] = None,
            y: Optional[str] = None,
            hover_data: Optional[Dict[str]] = None,
            size_default: Optional[int] = None,
            **kwargs,
    ):
        import plotly.express as px

        if df is None:
            df = self.get_data_frame_plot()

        if x is None:
            x = 'position_dim_0'

        if y is None:
            y = 'position_dim_1'

        if hover_data is None:
            hover_data = dict()
        hover_data['node'] = True
        hover_data['position_dim_0'] = False
        hover_data['position_dim_1'] = False

        if size_default is None:
            size_default = 50

        if df.index.name == 'node':
            df = df.reset_index()

        fig = (
            px.scatter(
                df,
                x=x,
                y=y,
                hover_data=hover_data,
                **kwargs,
            )
            .update_xaxes(visible=False, showgrid=False)
            .update_yaxes(visible=False, showgrid=False)
            .for_each_trace(
                lambda trace: trace.update(marker_size=size_default) if trace['marker_size'] is None else (),
            )
        )

        # for edge:
        # https://stackoverflow.com/questions/51410283/how-to-efficiently-create-interactive-directed-network-graphs-with-arrows-on-p

        return fig

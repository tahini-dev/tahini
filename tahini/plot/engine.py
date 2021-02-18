from __future__ import annotations
from typing import TYPE_CHECKING, Optional, Dict
from numbers import Number

from numpy import nan, sqrt
from pandas import DataFrame

if TYPE_CHECKING:
    from ..core.graph import TypeGraph
from ..core.edges import UndirectedEdges


class Base:

    def __init__(
            self,
            graph: TypeGraph,
    ):
        self.graph = graph

    def get_positions_nodes(
            self,
            positions_nodes: Optional[DataFrame] = None,
            **kwargs,
    ) -> DataFrame:

        if positions_nodes is None:
            positions_nodes = self.graph.nodes.update(data=self.graph.nodes.get_positions(**kwargs)).data

        if positions_nodes.index.name == self.graph.nodes.names_index[0]:
            positions_nodes = positions_nodes.reset_index()

        return positions_nodes

    def get_positions_edges(
            self,
            positions_edges: Optional[DataFrame] = None,
            positions_nodes: Optional[DataFrame] = None,
            **kwargs,
    ) -> DataFrame:

        if positions_edges is None:
            if positions_nodes is None:
                positions_nodes = self.graph.nodes.get_positions(**kwargs)

            positions_edges = self.graph.edges.update(
                data=self.graph.edges.get_positions(positions_nodes=positions_nodes),
            ).data

        if (
                self.graph.edges.names_index[0] in positions_edges.index.names and
                self.graph.edges.names_index[1] in positions_edges.index.names
        ):
            positions_edges = positions_edges.reset_index()

        return positions_edges

    def plot(self, **kwargs):
        ...


class Plotly(Base):

    # length units per size
    _ratio_length_size = .0565 / 20

    def plot(
            self,
            positions_nodes: Optional[DataFrame] = None,
            positions_edges: Optional[DataFrame] = None,
            size_node_default: Optional[Number] = None,
            arrow_edge_length: Optional[Number] = None,
            arrow_edge_angle: Optional[Number] = None,
            arrow_head: Optional[bool] = None,
            hover_data_nodes: Optional[Dict] = None,
            hover_data_edges: Optional[Dict] = None,
            kwargs_nodes: Optional[Dict] = None,
            kwargs_edges: Optional[Dict] = None,
            **kwargs,
    ):
        import plotly.express as px

        positions_nodes = self.get_positions_nodes(positions_nodes=positions_nodes, **kwargs)

        x = 'position_dim_0'
        y = 'position_dim_1'

        if hover_data_nodes is None:
            hover_data_nodes = dict()

        hover_data_nodes[self.graph.nodes.names_index[0]] = hover_data_nodes.get(self.graph.nodes.names_index[0], True)
        hover_data_nodes[x] = hover_data_nodes.get(x, False)
        hover_data_nodes[y] = hover_data_nodes.get(y, False)

        if size_node_default is None:
            size_node_default = 50

        if kwargs_nodes is None:
            kwargs_nodes = dict()
        kwargs_nodes['text'] = kwargs_nodes.get('text', self.graph.nodes.names_index[0])

        fig_nodes = (
            px.scatter(
                positions_nodes,
                x=x,
                y=y,
                hover_data=hover_data_nodes,
                **kwargs_nodes,
            )
            .update_xaxes(visible=False, showgrid=False)
            .update_yaxes(visible=False, showgrid=False)
            .for_each_trace(
                lambda t: t.update(marker_size=size_node_default) if t['marker_size'] is None else (),
            )
        )

        positions_edges = self.get_positions_edges(
            positions_edges=positions_edges,
            positions_nodes=positions_nodes,
            **kwargs,
        )

        diameter_node_default = size_node_default * self._ratio_length_size

        positions_edges = (
            positions_edges
            .assign(
                order_index=lambda df: df.index,
                length_raw=lambda df: sqrt(
                    (df['position_dim_0_end'] - df['position_dim_0_start']) ** 2 +
                    (df['position_dim_1_end'] - df['position_dim_1_start']) ** 2
                ),
                fraction_length=lambda df: 1 - diameter_node_default / df['length_raw'],
                skip_dim_0=lambda df: (df['position_dim_0_end'] - df['position_dim_0_start']) *
                                      (1 - df['fraction_length']),
                skip_dim_1=lambda df: (df['position_dim_1_end'] - df['position_dim_1_start']) *
                                      (1 - df['fraction_length']),
                position_dim_0_start=lambda df: df['position_dim_0_start'] + df['skip_dim_0'],
                position_dim_0_end=lambda df: df['position_dim_0_end'] - df['skip_dim_0'],
                position_dim_1_start=lambda df: df['position_dim_1_start'] + df['skip_dim_1'],
                position_dim_1_end=lambda df: df['position_dim_1_end'] - df['skip_dim_1'],
                position_dim_0_dummy=nan,
                position_dim_1_dummy=nan,
            )
            .drop(columns=['length_raw', 'skip_dim_0', 'fraction_length', 'skip_dim_1'])
        )

        if type(self.graph.edges) is UndirectedEdges:
            arrow_head = 0
        else:
            if arrow_head is None:
                arrow_head = 3

        annotations_fig = [
            dict(
                x=row['position_dim_0_end'],
                y=row['position_dim_1_end'],
                text='',
                showarrow=True,
                axref='x',
                ayref='y',
                ax=row['position_dim_0_start'],
                ay=row['position_dim_1_start'],
                arrowhead=arrow_head,
                arrowsize=4,
            )
            for _, row in positions_edges.iterrows()
        ]

        positions_edges = (
            positions_edges
            .melt(
                id_vars=[column for column in positions_edges.columns if 'position_dim_' not in column],
                value_vars=[column for column in positions_edges.columns if 'position_dim_' in column],
                var_name='position_type',
                value_name='position'
            )
            .assign(
                position_dim=lambda df: df['position_type'].str[:14],
                position_type=lambda df: df['position_type'].str[15:],
                order_position_type=lambda df: df['position_type'].map({
                    'start': 0,
                    'end': 1,
                    'dummy': 2,
                    'arrow_0_start': 3,
                    'arrow_0_end': 4,
                    'arrow_0_dummy': 5,
                    'arrow_1_start': 6,
                    'arrow_1_end': 7,
                    'arrow_1_dummy': 8,
                })
            )
        )

        positions_edges = (
            positions_edges
            .set_index(positions_edges.columns.drop('position').tolist())
            .unstack(level='position_dim')
            .droplevel(level=None, axis=1)
            .reset_index()
            .sort_values(['order_index', 'order_position_type'])
            .drop(columns=['order_index', 'order_position_type'])
        )

        if x not in positions_edges.columns and y not in positions_edges.columns:
            positions_edges = positions_edges.assign(**{
                x: nan,
                y: nan,
            })

        if hover_data_edges is None:
            hover_data_edges = dict()

        hover_data_edges[self.graph.edges.names_index[0]] = hover_data_edges.get(self.graph.edges.names_index[0], True)
        hover_data_edges[self.graph.edges.names_index[1]] = hover_data_edges.get(self.graph.edges.names_index[1], True)
        hover_data_edges[x] = hover_data_edges.get(x, False)
        hover_data_edges[y] = hover_data_edges.get(y, False)

        if kwargs_edges is None:
            kwargs_edges = dict()

        fig_edges = (
            px.line(
                positions_edges,
                x=x,
                y=y,
                hover_data=hover_data_edges,
                **kwargs_edges,
            )
            .update_xaxes(visible=False, showgrid=False)
            .update_yaxes(visible=False, showgrid=False)
            .for_each_trace(
                lambda t: t.update(line=dict(width=0)) if t['line']['width'] is None else (),
            )
        )

        fig = fig_nodes.update_layout(annotations=annotations_fig)

        for trace in fig_edges.data:
            fig = fig.add_trace(trace)

        return fig

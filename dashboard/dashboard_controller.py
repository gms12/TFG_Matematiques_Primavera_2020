from typing import Iterable, Dict, Optional, Callable
import logging

import dash
import dash_core_components as dcc
#import dash_bootstrap_components as dbc
import dash_html_components as html

from plotly.subplots import make_subplots
import plotly.graph_objs as go
import plotly.figure_factory as ff
import numpy as np

import torch

from dashboard.dashboard_data import DashboardData

TABS_STYLE = {
    'height': '44px',
    'width': '40%',
    'margin-left': '5%',
    # 'font-family': 'Helvetica',
    # 'font-size': '1.5em'
}
SUBTABS_STYLE = {
    'height': '35px',
    'width': '35%',
    'margin-left': '5%',
    # 'font-family': 'Helvetica',
    # 'font-size': '1.5em'
}

TAB_STYLE = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '6px',
    'fontWeight': 'bold',
}

SUBTAB_STYLE = {
    'borderBottom': '1px solid #d6d6d6',
    'padding': '4px',
    'fontWeight': 'bold',
    'margin-left': '4.8%',
}

TAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '6px'
}

SUBTAB_SELECTED_STYLE = {
    'borderTop': '1px solid #d6d6d6',
    'borderBottom': '1px solid #d6d6d6',
    'backgroundColor': '#119DFF',
    'color': 'white',
    'padding': '4px'
}

COLOR_TIME = {
    'total_time': '#023858',
    'sma': '#d73027',
    'ema': '#fdae61',
    'loading_time': '#d73027',
    'fitting_time': '#fc8d59',
    'prediction_time': '#fee090',
    'business_time': '#abd9e9',
    'cosmos_time': '#74add1',
    'hive_time': '#4575b4',
}

class DashboardController():
    """
    Controller class that handles the Dashboard web app.
    """

    def __init__(self, data: DashboardData) -> None:
        """Constructor method for the DashboardController class.

        Parameters
        ----------
        data: dahsboard.DashboardData
            Object that handles the data to be displayed in the dashboard

        """
        self._data = data

    @staticmethod
    def _style_right(size: float = 45) -> Dict:
        """This method returns a css style for a div placed on the right. Custom width.

        Parameters
        ----------
        size: float

        Returns
        -------
        Dict
            Dictionary containing the specifications of the style.

        """
        return {
            'width': '{}%'.format(size),
            'display': 'inline-block',
            'margin-right': '5%'
        }

    @staticmethod
    def _style_left(size: float = 45) -> Dict:
        """This method returns a css style for a div placed on the left. Custom width.

        Parameters
        ----------
        size: float

        Returns
        -------
        Dict
            Dictionary containing the specifications of the style.

        """
        return {
            'width': '{}%'.format(size),
            'display': 'inline-block',
            'margin-left': '5%',
        }

    @staticmethod
    def _style_middle(size: float = 45) -> Dict:
        """This method returns a css style for a div placed on the left. Custom width.

        Parameters
        ----------
        size: float

        Returns
        -------
        Dict
            Dictionary containing the specifications of the style.

        """
        return {
            'width': '{}%'.format(size),
            'display': 'inline-block',
            # 'margin-left': '5%',
        }

    def serve_layout(self) -> html.Div:
        """Layout of the Dashboard.

        Returns
        -------
         dash_html_components.Div
            HTML Div component containing the layout of the web app.
        """
        layout = html.Div(children=[
            html.H1(children='Latent Space Explorer',
                    style={'margin-left': '5%'}),
            dcc.RadioItems(id='neurons', options=[
                    {'label': '1 neuron', 'value': 1},
                    {'label': '2 neurons', 'value': 2},
                    {'label': '3 neurons', 'value': 3}
                ], value=1, labelStyle={'display': 'inline-block'}, style={'margin-left': '5%'}),
            dcc.Tabs(id='tabs', value='norm-tab', children=[
                dcc.Tab(label='Normalisation', value='norm-tab', children=[
                ], style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE),
                dcc.Tab(label='Standardisation', value='st-tab', children=[
                ],
                style=TAB_STYLE, selected_style=TAB_SELECTED_STYLE)
            ], style=TABS_STYLE),


            html.Div(children=[
                dcc.Graph(id='ls-plot'),
                html.Div(id='x-lbl', children='x: ', style=self._style_left(5)),
                html.Div(id='x-sl-div',children=[dcc.Slider(id='x-slider', step=0.05, value=0)], style=self._style_middle(70)),
                html.Div(id='x-val', children='     x', style=self._style_right(15)),
                html.Div(id='y-lbl',children='y: ', style=self._style_left(5)),
                html.Div(id='y-sl-div', children=[dcc.Slider(id='y-slider', step=0.05, value=0)], style=self._style_middle(70)),
                html.Div(id='y-val', children='     x', style=self._style_right(15)),
                html.Div(id='z-lbl',children='z: ', style=self._style_left(5)),
                html.Div(id='z-sl-div',children=[dcc.Slider(id='z-slider', step=0.05, value=0)], style=self._style_middle(70)),
                html.Div(id='z-val', children='     x', style=self._style_right(15)),
            ], style=self._style_left(45)),

            html.Div(children=[
                dcc.Graph(id='pred-plot'),
                html.Div(children='d: ', style=self._style_left(5)),
                html.Div([dcc.Slider(id='d-slider', min=0, max=0.5, step=0.01, value=0)], style=self._style_middle(70)),
                html.Div(id='d-val', children='     x', style=self._style_right(15)),
            ], style=self._style_right(45)),

            dcc.Interval(id='interval-component', interval=1000000, n_intervals=0),
            html.Div(id='data-update-div', style={'visibility': 'hidden'})
        ])
        return layout

    def update_hidden_space(self, n: int, tr_type: str, x, y, z) -> go.Figure:
        if tr_type == 'st-tab':
            H = self._data.hidden_st[n]
            nd = self._data.nd_st[n]
        else:
            H = self._data.hidden_mm[n]
            nd = self._data.nd_mm[n]

        if n == 1:
            fig = go.Figure(data=[
                go.Scatter(x=H[:, 0], y=[0 for i in range(H.shape[0])], mode='markers',
                marker=dict(size=7, opacity=0.2, color=nd, colorscale='Jet',
                    #colorbar=dict(title='Density of Neighbors'),
                )),
                go.Scatter(x=[x], mode='markers',
                    marker=dict(size=20, color='black', opacity=0.7))],
            )
        elif n == 2:
            fig = go.Figure(data=[
                go.Scatter(x=H[:, 0], y=H[:, 1], mode='markers',
                marker=dict(size=7, opacity=0.2, color=nd, colorscale='Jet',
                    #colorbar=dict(title='Density of Neighbors'),
                )),
                go.Scatter(x=[x], y=[y], mode='markers',
                    marker=dict(size=20, color='black', opacity=0.7))],
            )
        else:
            fig = go.Figure(data=[
                go.Scatter3d(x=H[:, 0], y=H[:, 1], z=H[:, 2], mode='markers',
                    marker=dict(size=3, opacity=0.2, color=nd, colorscale='Jet',
                        #colorbar=dict(title='Density of Neighbors'),
                    )),
                go.Scatter3d(x=[x], y=[y], z=[z], mode='markers',
                 marker=dict(size=10, color='black', opacity=0.7))],
                 layout=go.Layout(plot_bgcolor='rgba(0, 0, 0, 0)')
            )
            fig.update_layout(margin=dict(l=0, r=0, b=0, t=0))
        fig.update_layout(showlegend=False, xaxis_showticklabels=False, yaxis_showticklabels=False)
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        return fig

    def update_pred(self, n: int, tr_type: str, x, y, z, d):
        if tr_type == 'st-tab':
            model = self._data.models_st[n]
            H = self._data.hidden_st[n]

            data = self._data.standard_data
        else:
            model = self._data.models_mm[n]
            H = self._data.hidden_mm[n]

            data = self._data.normalized_data

        if n == 1:
            p = np.array([x], dtype=np.float16)

            point = torch.from_numpy(p).type(torch.FloatTensor)

            delta_x = torch.from_numpy(np.array([d])).type(torch.FloatTensor)
            pred_point = model.decode(point).clone().detach().numpy()

            fig = go.Figure(data=[
                go.Scatter(y=model.decode(point).clone().detach().numpy(), line=dict(width=4, color='black', dash='dash'), name='Selected Point'),
                go.Scatter(y=model.decode(point + delta_x).clone().detach().numpy(), line=dict(width=2, color='#7fc97f', dash='dash'), name='+ delta_x'),
                go.Scatter(y=model.decode(point - delta_x).clone().detach().numpy(), line=dict(width=2, color='#7fc97f'), name='- delta_x')
            ], layout=go.Layout(width=700, height=500, showlegend=True, legend_orientation="h"))

        elif n == 2:
            p = np.array([x, y], dtype=np.float16)

            point = torch.from_numpy(p).type(torch.FloatTensor)
            delta_x = torch.from_numpy(np.array([d, 0])).type(torch.FloatTensor)
            delta_y = torch.from_numpy(np.array([0, d])).type(torch.FloatTensor)


            fig = go.Figure(data=[
                go.Scatter(y=model.decode(point).clone().detach().numpy(), line=dict(width=4, color='black', dash='dash'), name='Selected Point'),
                go.Scatter(y=model.decode(point + delta_x).clone().detach().numpy(), line=dict(width=2, color='#7fc97f', dash='dash'), name='+ delta_x'),
                go.Scatter(y=model.decode(point - delta_x).clone().detach().numpy(), line=dict(width=2, color='#7fc97f'), name='- delta_x'),
                go.Scatter(y=model.decode(point + delta_y).clone().detach().numpy(), line=dict(width=2, color='#beaed4', dash='dash'), name='+ delta_y'),
                go.Scatter(y=model.decode(point - delta_y).clone().detach().numpy(), line=dict(width=2, color='#beaed4'), name='- delta_y'),
                ], layout=go.Layout(width=700, height=500, showlegend=True, legend_orientation="h"))

        else:
            p = np.array([x, y, z], dtype=np.float16)
            point = torch.from_numpy(p).type(torch.FloatTensor)
            delta_x = torch.from_numpy(np.array([d, 0, 0])).type(torch.FloatTensor)
            delta_y = torch.from_numpy(np.array([0, d, 0])).type(torch.FloatTensor)
            delta_z = torch.from_numpy(np.array([0, 0, d])).type(torch.FloatTensor)

            fig = go.Figure(data=[
                go.Scatter(y=model.decode(point).clone().detach().numpy(), line=dict(width=4, color='black', dash='dash'), name='Selected Point'),
                go.Scatter(y=model.decode(point + delta_x).clone().detach().numpy(), line=dict(width=2, color='#7fc97f', dash='dash'), name='+ delta_x'),
                go.Scatter(y=model.decode(point - delta_x).clone().detach().numpy(), line=dict(width=2, color='#7fc97f'), name='- delta_x'),
                go.Scatter(y=model.decode(point + delta_y).clone().detach().numpy(), line=dict(width=2, color='#beaed4', dash='dash'), name='+ delta_y'),
                go.Scatter(y=model.decode(point - delta_y).clone().detach().numpy(), line=dict(width=2, color='#beaed4'), name='- delta_y'),
                go.Scatter(y=model.decode(point + delta_z).clone().detach().numpy(), line=dict(width=2, color='#f768a1', dash='dash'), name='+ delta_z'),
                go.Scatter(y=model.decode(point - delta_z).clone().detach().numpy(), line=dict(width=2, color='#f768a1'), name='- delta_z'),
                ],
                layout=go.Layout(width=700, height=500, showlegend=True, legend_orientation="h"))

        fig.update_layout(xaxis_showticklabels=False, yaxis_showticklabels=False)
        fig.update_layout({'plot_bgcolor': 'rgba(0, 0, 0, 0)', 'paper_bgcolor': 'rgba(0, 0, 0, 0)',
        })
        return fig

    def update_x_slider(self, n: int, tr_type: str):
        if tr_type == 'st-tab':
            H = self._data.hidden_st[n]
        else:
            H = self._data.hidden_mm[n]
        min_val = min(H[:, 0])
        max_val = max(H[:, 0])
        logging.info(min_val)
        logging.info(max_val)
        logging.info((max_val - min_val)/2)
        return [min_val, max_val, (max_val + min_val)/2]

    def update_y_slider(self, n: int, tr_type: str):
        if n < 2:
            return [dash.no_update, dash.no_update, dash.no_update]

        if tr_type == 'st-tab':
            H = self._data.hidden_st[n]
        else:
            H = self._data.hidden_mm[n]
        min_val = min(H[:, 1])
        max_val = max(H[:, 1])
        return [min_val, max_val, (max_val + min_val)/2]

    def update_z_slider(self, n: int, tr_type: str):
        if n < 3:
            return [dash.no_update, dash.no_update, dash.no_update]

        if tr_type == 'st-tab':
            H = self._data.hidden_st[n]
        else:
            H = self._data.hidden_mm[n]
        min_val = min(H[:, 2])
        max_val = max(H[:, 2])
        return [min_val, max_val, (max_val + min_val)/2]

    def update_y_slider_style(self, n: int):
        if n < 2:
            return [{'visibility': 'hidden'} for i in range(3)]
        return [self._style_left(5), self._style_middle(70), self._style_right(15)]

    def update_z_slider_style(self, n: int):
        if n < 3:
            return [{'visibility': 'hidden'} for i in range(3)]
        return [self._style_left(5), self._style_middle(70), self._style_right(15)]

    def update_slider_val(self, x, y, z, d):
        return ['{:.2f}'.format(x), '{:.2f}'.format(y), '{:.2f}'.format(z), d]

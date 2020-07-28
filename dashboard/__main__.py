import argparse
import os
import sys
import logging

import dash
from dash.dependencies import Input, Output, State

from dashboard.dashboard_data import DashboardData
from dashboard.dashboard_controller import DashboardController

logging.getLogger().setLevel(logging.INFO)
# TODO: change country choices
def main():
    try:
        external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
        app = dash.Dash(__name__, external_stylesheets=external_stylesheets, )

        data = DashboardData()

        controller = DashboardController(data=data)

        app.layout = controller.serve_layout

        app.callback(Output('ls-plot', 'figure'),
                     [Input('neurons', 'value'), Input('tabs', 'value'),
                      Input('x-slider', 'value'), Input('y-slider', 'value'),
                      Input('z-slider', 'value')])(controller.update_hidden_space)

        app.callback([Output('x-slider', 'min'), Output('x-slider', 'max'),
                        Output('x-slider', 'value')],
                     [Input('neurons', 'value'),
                     Input('tabs', 'value')])(controller.update_x_slider)

        app.callback([Output('y-slider', 'min'), Output('y-slider', 'max'),
                        Output('y-slider', 'value')],
                     [Input('neurons', 'value'),
                     Input('tabs', 'value')])(controller.update_y_slider)

        app.callback([Output('z-slider', 'min'), Output('z-slider', 'max'),
                        Output('z-slider', 'value')],
                     [Input('neurons', 'value'),
                     Input('tabs', 'value')])(controller.update_z_slider)

        app.callback(Output('pred-plot', 'figure'),
                     [Input('neurons', 'value'), Input('tabs', 'value'),
                      Input('x-slider', 'value'), Input('y-slider', 'value'),
                      Input('z-slider', 'value'),
                      Input('d-slider', 'value')])(controller.update_pred)

        app.callback([Output('y-lbl', 'style'), Output('y-sl-div', 'style'),
                        Output('y-val', 'style')],
                     [Input('neurons', 'value')])(controller.update_y_slider_style)

        app.callback([Output('z-lbl', 'style'), Output('z-sl-div', 'style'),
                        Output('z-val', 'style')],
                     [Input('neurons', 'value')])(controller.update_z_slider_style)

        app.callback([Output('x-val', 'children'), Output('y-val', 'children'),
                      Output('z-val', 'children'), Output('d-val', 'children'),],
                     [Input('x-slider', 'value'), Input('y-slider', 'value'),
                      Input('z-slider', 'value'),
                      Input('d-slider', 'value')])(controller.update_slider_val)

        app.run_server(debug=True)
        # app.run_server(debug=False, host='0.0.0.0')
    except Exception as e:
        logging.critical(e)
        sys.exit(0)

if __name__ == '__main__':
    main()

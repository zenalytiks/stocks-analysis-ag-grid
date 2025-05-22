import dash
from dash import html, dcc
import dash_ag_grid as dag
import dash_bootstrap_components as dbc

dash.register_page(__name__, path='/', name='Stockmarket Dashboard')

def layout():
    return dbc.Row(
                [
                    dbc.Col(
                        [
                            html.Div(id='hidden-div',style={'display':'none'}),
                            dbc.Stack(dbc.Button("Refresh",id='refresh-table',n_clicks=0,color='primary',className='ms-auto'),direction='horizontal',className='mb-3'),
                            dcc.Loading(html.Div(
                                [
                                    dag.AgGrid(
                                        id="portfolio-grid",
                                        className="ag-theme-alpine-dark",
                                        columnSize="responsiveSizeToFit",
                                        style={"height": "600px", "width": "100%"},
                                        dashGridOptions={
                                            "animateRows": False,
                                            "loadingOverlayComponent": "CustomLoadingOverlay",
                                            "loadingOverlayComponentParams": {
                                                "loadingMessage": "",
                                                "color": "red",
                                            }
                                        },
                                    )
                                ],className="dashboard-container"
                            ),overlay_style={"visibility":"visible", "filter": "blur(2px)"})
                        ],className='p-5'
                    )
                ]
            )
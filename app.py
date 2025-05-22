import dash
from dash import html,dcc
import dash_bootstrap_components as dbc
from config import app
import callbacks

server = app.server

app.layout = dbc.Container(
    [
        dcc.Location(id='url',pathname='/'),
        html.H1("Stockmarket Dashboard",className='text-center p-3'),
        dash.page_container
    ],fluid=True
)


if __name__ == "__main__":
    server.run(debug=False)
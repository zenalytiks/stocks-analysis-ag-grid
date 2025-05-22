from dash import Input, Output, callback, no_update, ctx
from helpers import generate_table, write_comments_to_json

@callback(
    [
        Output('portfolio-grid','rowData'),
        Output('portfolio-grid','columnDefs'),
        Output('portfolio-grid','defaultColDef')
    ],
    [
        Input('url','pathname'),
        Input('refresh-table','n_clicks')
    ]
)
def update_table(pathname,n_clicks):
    if pathname == "/" or ctx.triggered_id == 'refresh-table':
        rowData,columnDefs,defaultColDef = generate_table()

        return [rowData,columnDefs,defaultColDef]
    else:
        return [no_update] * 3
    
@callback(
    [
        Output('hidden-div','children')
    ],
    [
        Input('url','pathname'),
        Input('portfolio-grid','cellValueChanged')
    ]
)

def update_comments(pathname,cell_value):
    if pathname == "/" and cell_value:

        write_comments_to_json("./comments.json", {cell_value[0]['data']['ticker'] : cell_value[0]['data']['comments']})
        return [no_update]
    else:
        return [no_update]
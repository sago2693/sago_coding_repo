import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

from app import app
from apps import app1, app2

#Main layout

url_bar_and_content_div = html.Div([
    dcc.Location(id='url', refresh=False),
    html.Div(id='page-content')
])

# index layout
app.layout = url_bar_and_content_div

# "complete" layout to supress callback exceptions
app.validation_layout = html.Div([
    url_bar_and_content_div,
    app1.layout,
    app2.layout
])

@app.callback(Output('page-content', 'children'),
              [Input('url', 'pathname')])
def display_page(pathname):
    if pathname == '/':
        return 'Home'
    if pathname == '/apps/app1':
        return app1.layout
    elif pathname == '/apps/app2':
        return app2.layout
    else:
        return '404'

if __name__ == '__main__':
    app.run_server(debug=True)
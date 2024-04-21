import dash_bootstrap_components as dbc

from dash import Input, Output, dcc, html
from layouts.about_page import NavBar_Layout, UI_Content_Layout, Dashboard_Layout, EDA_Layout, Model_Training_Layout

from server import app


dcc.Store(id='session', storage_type='session')
app.title = "SpeakSense - AI Language Detection Tool"

app.layout = html.Div(
    [
        html.Div([NavBar_Layout]),

        dbc.Container(
            children=[
                dcc.Location(id="url"),
                UI_Content_Layout,
            ],
            fluid=True, className="p-5 dashboard-main-wrapper",
        )
    ], id='app-layout-body'
)

# --------------------------------------------- ## CALLBACK METHODS ## --------------------------------------------- #


@app.callback(
    Output("page-content", "children"),
    [
        Input("url", "pathname")
    ]
)
def render_page_content(path_name: str) -> object:
    """Renders Page Contents based on the URL Path.

    Args:
        path_name (str): Current URL Path of the Web Application.

    Returns:
        object: Page Content Layout
    """

    if path_name == "/":
        return Dashboard_Layout

    elif path_name == "/eda-page":
        return EDA_Layout

    elif path_name == "/model-training-page":
        return Model_Training_Layout

    # elif path_name == "/classification-details":
    #     return Classification_Layout

    return html.Div(
        children=[
            html.H1("404: Not found", className="text-danger"),
            html.Hr(),
            html.P(f"The pathname {path_name} was not recognised..."),
        ], className="p-3 bg-light rounded-3",
    )


if __name__ == "__main__":
    # app.run_server(debug=True, dev_tools_ui=False,
    #    host='0.0.0.0', dev_tools_props_check=False)

    app.run_server(debug=True)

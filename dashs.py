import os
from embedding import generate_embeddings
from options.options import Options
import plotly.express as px
import numpy as np
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
from dash.exceptions import PreventUpdate
import pandas as pd
import plotly.graph_objs as go

opt = Options()

if os.path.exists("data/embeddings.csv"):
    print("Data already generated. Loading data...")
    data = pd.read_csv("data/embeddings.csv")
else:
    print("Generating data...")
    datafile, label_images = generate_embeddings()
    data = datafile.dataset.data

with open("dash_files/desc.md", "r") as file:
    desc_md = file.read()
    
static = "/static/"

def create_layout(app, dataset = data):
    def embedding_scatter_plot(df):
        try:  
            fig = px.scatter_3d(data, x="embeddings_x", y="embeddings_y", z="embeddings_z", color = "labels")
            return fig
        except KeyError as error:
            print(error)
            raise PreventUpdate

    def display_3d_scatter_plot(dataset):
        if not dataset.empty:
            figure = embedding_scatter_plot(dataset)

        else:
            print("Empty Dataset...")
            figure = go.Figure()

        return figure
    return html.Div(
        className="row",
        style={"max-width": "100%", "font-size": "1.5rem", "padding": "0px 0px"},
        children=[
            # Header
            html.Div(
                className="row header",
                id="app-header",
                style={"background-color": "#f9f9f9"},
                children=[
                    html.Div(
                        [
                            html.H3(
                                "Autoencoder File Sorter",
                                className="header_title",
                                id="app-title",
                            )
                        ],
                        className="nine columns header_title_container",
                    ),
                ],
            ),
            # Demo Description
            html.Div(
                className="row background",
                id="demo-explanation",
                style={"padding": "50px 45px"},
                children=[
                    html.Div(
                        id="description-text", children=dcc.Markdown(desc_md)
                    )
                ],
            ),
            # Body
            html.Div(
                className="row_background",
                style={"padding": "10px"},
                children=[
                    html.Div(className="nine columns",
                             children = [
                                 dcc.Graph(id="graph-3d-plot-embedding",
                                      figure=display_3d_scatter_plot(dataset)
                                )
                            ]
                    ),
                    html.Div(
                        className="three columns",
                        id="euclidean-distance",
                        children=[
                            html.Section(className = "card-style",
                                children=[
                                    html.Div(
                                        id="div-plot-click-message",
                                        style={
                                            "text-align": "center",
                                            "margin-bottom": "7px",
                                            "font-weight": "bold",
                                        },
                                    ),
                                    html.Img(id="div-plot-click-image", 
                                        height = 480, width = 720)
                                ],
                                style={"padding": "5px"}
                            )
                        ],
                    )
                ], 
            ),
        ],
    )


def callbacks(app, dataset = data):
    
    @app.callback(
        Output("div-plot-click-image", "src"),
        [Input("graph-3d-plot-embedding", "clickData")],
    )
    def display_click_image(clickData, app = app, dataset = dataset):
        if clickData:
            # Convert the point clicked into float64 numpy array
            click_point_np = np.array(
                [clickData["points"][0][i] for i in ["x", "y", "z"]]).astype(np.float64)
            # Create a boolean mask of the point clicked, truth value exists at only one row
            bool_mask_click = (dataset.loc[:, "embeddings_x":"embeddings_z"].eq(click_point_np).all(axis=1))
            # Retrieve the index of the point clicked, given it is present in the set
            if bool_mask_click.any():
                clicked_idx = dataset[bool_mask_click].index[0]

                # Retrieve the image corresponding to the index
                image_path = dataset["path"].iloc[clicked_idx]
                # removing filepath
                filename = image_path.replace(opt.filepath, "")
                
                return app.get_asset_url(filename)
        return None
    

    @app.callback(
        Output("div-plot-click-message", "children"),
        [Input("graph-3d-plot-embedding", "clickData")],
    )
    def display_click_message(clickData, dataset = dataset):
        # Displays message shown when a point in the graph is clicked
        if clickData:
            return "Image Selected"
        else:
            return "Click a data point on the scatter plot to display its corresponding image."


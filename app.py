# Libraries
import pandas as pd
import os
import string
import numpy as np

import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

from dash import Dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

### Data preparation 
# Upload the datasets you need & save
tfidf_rank = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/dashboard_data.csv', low_memory=False)


markdown_text = """


"""

### Scatterplot
fig_1 = px.scatter(tfidf_rank, 'term_rank', 'TF-IDF', hover_name='sub_indicator', hover_data=['TF-IDF', 'TF-IDF_Z'], log_x = True, height=600)
fig_1.layout.width = 1400
fig_1.layout.height = 650

### Create app
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    [
        html.H1("InnovateEDU", style={'textAlign': 'center'}),
        
        dcc.Markdown(children = markdown_text),
        
        html.H4("TFIDF and term rank for Unigrams in WWC corpus", ),
        
        dcc.Graph(figure=fig_1)
        
        
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)

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
unigrams = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/unigram_tfidf_scores_math.csv', low_memory=False)
unigrams_keywords = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/unigrams_keywords_math.csv', low_memory=False)

markdown_text = """


"""

### Scatterplot
fig_1 = px.scatter(tfidf_rank, 'term_rank', 'TF-IDF', hover_name='sub_indicator', hover_data=['TF-IDF', 'TF-IDF_Z'], log_x = True)
fig_1.layout.width = 1280
fig_1.layout.height = 650

### Horizontal Bar Charts
unigrams_top30 = unigrams.sort_values('TF-IDF_Z', ascending=False).head(30)
unigrams_keywords_top30 = unigrams_keywords.sort_values('TF-IDF_Z', ascending=False).head(30)
fig_2 = px.bar(unigrams_top30, x='sub_indicator', y='TF-IDF_Z', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_3 = px.bar(unigrams_keywords_top30, x='sub_indicator', y='TF-IDF_Z', hover_data=['TF-IDF', 'TF-IDF_Z'])
### Create app
app = Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

app.layout = html.Div(
    [
        html.H1("InnovateEDU Interactive DashBoard", style={'textAlign': 'center'}),
        
        dcc.Markdown(children = markdown_text),
        
        html.H4("TFIDF and Term Rank for Unigrams in WWC corpus" ),
        
        dcc.Graph(figure=fig_1),
        
        html.Div([
            
            html.H4("Top 30 Unigrams in WWC Math Corpus"),
            
            dcc.Graph(figure=fig_2)], style = {'width':'50%', 'float':'left'}),
        
        html.Div([
            
            html.H4("Top 30 Unigram Indicators in WWC Math Corpus"),
            
            dcc.Graph(figure=fig_3)], style = {'width':'50%', 'float':'right'}),
        
        
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)

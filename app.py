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
unigrams_math = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/unigram_tfidf_scores_math.csv', low_memory=False)
unigrams_keywords_math = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/unigrams_keywords_math.csv', low_memory=False)
bigrams_math = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/bigram_tfidf_scores_math.csv', low_memory=False)
bigrams_keywords_math = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/bigrams_keywords_math.csv', low_memory=False)
alldata_math = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/all_tfidf_math.csv', low_memory=False)
alldata_keywords_math = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/all_tfidf_keywords_math.csv', low_memory=False)

unigrams_reading = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/unigram_tfidf_scores_reading.csv', low_memory=False)
unigrams_keywords_reading = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/unigrams_keywords_reading.csv', low_memory=False)
bigrams_reading = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/bigram_tfidf_scores_reading.csv', low_memory=False)
bigrams_keywords_reading = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/bigrams_keywords_reading.csv', low_memory=False)
alldata_reading = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/alldata_tfidf_reading.csv', low_memory=False)
alldata_keywords_reading = pd.read_csv('https://github.com/haizhuhong/capstone-dashboard/raw/main/alldata_keywords_reading.csv', low_memory=False)

markdown_text = """


"""

### Scatterplot
fig_1 = px.scatter(tfidf_rank, 'term_rank', 'TF-IDF', hover_name='sub_indicator', hover_data=['TF-IDF', 'TF-IDF_Z'], log_x = True)
fig_1.layout.width = 1280
fig_1.layout.height = 650

### Horizontal Bar Charts
## Math
#Unigrams
fig_2 = px.bar(unigrams_math, x='TF-IDF_Z', y='sub_indicator', orientation='h', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_3 = px.bar(unigrams_keywords_math, x='TF-IDF_Z', y='sub_indicator', orientation='h',hover_data=['TF-IDF', 'TF-IDF_Z'])
#Bigrams
fig_4 = px.bar(bigrams_math, x='TF-IDF_Z', y='sub_indicator', orientation='h', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_5 = px.bar(bigrams_keywords_math, x='TF-IDF_Z', y='sub_indicator', orientation='h',hover_data=['TF-IDF', 'TF-IDF_Z'])
#All_data
fig_6 = px.bar(alldata_math, x='TF-IDF_Z', y='sub_indicator', orientation='h', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_7 = px.bar(alldata_keywords_math, x='TF-IDF_Z', y='sub_indicator', orientation='h',hover_data=['TF-IDF', 'TF-IDF_Z'])

## Reading
#Unigrams
fig_8 = px.bar(unigrams_reading, x='TF-IDF_Z', y='sub_indicator', orientation='h', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_9 = px.bar(unigrams_keywords_reading, x='TF-IDF_Z', y='sub_indicator', orientation='h',hover_data=['TF-IDF', 'TF-IDF_Z'])
#Bigrams
fig_10 = px.bar(bigrams_reading, x='TF-IDF_Z', y='sub_indicator', orientation='h', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_11 = px.bar(bigrams_keywords_reading, x='TF-IDF_Z', y='sub_indicator', orientation='h',hover_data=['TF-IDF', 'TF-IDF_Z'])
#All_data
fig_12 = px.bar(alldata_reading, x='TF-IDF_Z', y='sub_indicator', orientation='h', hover_data=['TF-IDF', 'TF-IDF_Z'])
fig_13 = px.bar(alldata_keywords_reading, x='TF-IDF_Z', y='sub_indicator', orientation='h',hover_data=['TF-IDF', 'TF-IDF_Z'])

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
        
        html.Div([
            
            html.H4("Top 30 Bigrams in WWC Math Corpus"),
            
            dcc.Graph(figure=fig_4)], style = {'width':'50%', 'float':'left'}),
        
        html.Div([
            
            html.H4("Top 30 Bigram Indicators in WWC Math Corpus"),
            
            dcc.Graph(figure=fig_5)], style = {'width':'50%', 'float':'right'}),
        
        html.Div([
            
            html.H4("Top 30 N-Grams in WWC Math Corpus"),
            
            dcc.Graph(figure=fig_6)], style = {'width':'50%', 'float':'left'}),
        
        html.Div([
            
            html.H4("Top 30 N-Gram Indicators in WWC Math Corpus"),
            
            dcc.Graph(figure=fig_7)], style = {'width':'50%', 'float':'right'}),
        
        html.Div([
            
            html.H4("Top 30 Unigrams in WWC Reading Corpus"),
            
            dcc.Graph(figure=fig_8)], style = {'width':'50%', 'float':'left'}),
        
        html.Div([
            
            html.H4("Top 30 Unigram Indicators in WWC Reading Corpus"),
            
            dcc.Graph(figure=fig_9)], style = {'width':'50%', 'float':'right'}),
        
        html.Div([
            
            html.H4("Top 30 Bigrams in WWC Reading Corpus"),
            
            dcc.Graph(figure=fig_10)], style = {'width':'50%', 'float':'left'}),
        
        html.Div([
            
            html.H4("Top 30 Bigram Indicators in WWC Reading Corpus"),
            
            dcc.Graph(figure=fig_11)], style = {'width':'50%', 'float':'right'}),
        
        html.Div([
            
            html.H4("Top 30 N-Grams in WWC Reading Corpus"),
            
            dcc.Graph(figure=fig_12)], style = {'width':'50%', 'float':'left'}),
        
        html.Div([
            
            html.H4("Top 30 N-Gram Indicators in WWC Reading Corpus"),
            
            dcc.Graph(figure=fig_13)], style = {'width':'50%', 'float':'right'}),
        
            
    ]
)


if __name__ == '__main__':
    app.run_server(debug=True)

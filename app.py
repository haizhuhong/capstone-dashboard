# Libraries
import gensim
import pandas as pd
import os
import string
from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn import preprocessing

import plotly.graph_objects as go
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

import dash
from jupyter_dash import JupyterDash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import nltk
import string

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

### Data preparation 
# Set working directory to where all the datasets are 
os.chdir('/Users/haizhuhong/Documents/UVA DataScience/Spring2021/Capstone/Semester2-main/datasets')
# Upload the datasets you need & save
data = pd.read_csv ('Math_WWC_with_EricID.csv')

# Subset for the only two columns we'll analyze
data2 = data[['title', 'description']]
data2.head(2)

# Combine title + abstract (description) into one box per paper
data_3 = data["title"] + " " + data["description"]
data_3.head(3)

# Take out punctuation

for i in range(len(data_3)):
    data_3[i] = data_3[i].translate(str.maketrans('', '', string.punctuation)) 
    
# Remove stop words 
# This step also removes punctuation, so we don't need an extra function like our R code has
from gensim.parsing.preprocessing import remove_stopwords
for i in range(len(data_3)):
    data_3[i] = remove_stopwords(data_3[i].lower())
    
# Tokenize with a simple list comprehension
# Puts every text box into a pile and tokenizes into one huge string of words
xx = [text.split() for text in data_3] 

# Flatten the nested list into one single list
xx_flat = [item for sublist in xx for item in sublist]

# NOTE: This resulting xx vector should be similar to the xx vector we have in R  

# Lemmatize xx_flat
# nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer 

# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()

# Lemmatize
lemmatized_output = [lemmatizer.lemmatize(w) for w in xx_flat]
xx_flat = lemmatized_output

# Takes a count of all words that show up 
unigram = Counter(xx_flat)
unigram_df = pd.DataFrame.from_dict(unigram, orient='index', columns=['freq_count'])
unigram_df.index.names = ['unigram']
unigram_data = unigram_df.sort_values('freq_count', ascending=False)

corpus = list(data_3)
# Init the Wordnet Lemmatizer
lemmatizer = WordNetLemmatizer()
w_tokenizer = nltk.tokenize.WhitespaceTokenizer()

# Lemmatize
def lemmatize_text(text):
    return [lemmatizer.lemmatize(w.lower()) for w in w_tokenizer.tokenize(text)]
lemmatized_output = data_3.apply(lemmatize_text)
data_3 = lemmatized_output.apply(' '.join)

# instantiate the vectorizer object
countvectorizer = CountVectorizer(analyzer= 'word', stop_words='english')
tfidfvectorizer = TfidfVectorizer(analyzer='word',stop_words= 'english')

# convert the documents into a matrix
count_wm = countvectorizer.fit_transform(data_3)
tfidf_wm = tfidfvectorizer.fit_transform(data_3)

# retrieve the terms found in the corpora
# - if we take same parameters on both Classes(CountVectorizer and TfidfVectorizer),
# - it will give same output of get_feature_names() methods)
#count_tokens = tfidfvectorizer.get_feature_names() # no difference
count_tokens = countvectorizer.get_feature_names()
tfidf_tokens = tfidfvectorizer.get_feature_names()
df_countvect = pd.DataFrame(data = count_wm.toarray(),columns = count_tokens)
df_tfidfvect = pd.DataFrame(data = tfidf_wm.toarray(),columns = tfidf_tokens)
df = pd.DataFrame(df_tfidfvect)

means_tfidf = pd.DataFrame(df.mean())
means_tfidf.columns = ['TF-IDF'] 
means_tfidf['TF-IDF_Z'] = (means_tfidf - means_tfidf.mean())/means_tfidf.std()
means_tfidf["sub_indicator"] = df.columns
unigrams_tfidf = means_tfidf.sort_values('TF-IDF_Z', ascending=False)
unigram_data = unigram_data.reset_index().rename(columns={'unigram': 'sub_indicator'})
tfidf_rank = pd.merge(unigrams_tfidf, unigram_data, how='outer').fillna(0.0).sort_values("freq_count", ascending=False)
tfidf_rank['term_rank'] = [r+1 for r in range(tfidf_rank.shape[0])]


markdown_text = """

Interactive Scatterplot for WWC Corpus TFIDF Results.

"""

### Scatterplot
fig_1 = px.scatter(tfidf_rank, 'term_rank', 'TF-IDF', hover_name='sub_indicator', hover_data=['TF-IDF', 'TF-IDF_Z'], log_x = True, height=600)
fig_1.layout.width = 700
fig_1.layout.height = 650

### Create app
app = JupyterDash(__name__, external_stylesheets=external_stylesheets)
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

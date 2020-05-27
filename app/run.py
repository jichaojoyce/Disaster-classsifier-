import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag, word_tokenize
import nltk

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
from sklearn.base import BaseEstimator, TransformerMixin

app = Flask(__name__)

class StartingVerbExtractor(BaseEstimator, TransformerMixin):
    """
    Starting Verb Extractor class
    
    This class extract the starting verb of a sentence, aim to creat a new feature for the ML classifier
    Parameters:
        text -> list of tokenized text messages
    Output:
        pd.DataFrame(X_tagged)-> a dataframe with the starting verb of a sentence
    """
    def starting_verb(self, text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            first_word, first_tag = pos_tags[0]
            if first_tag in ['VB', 'VBP'] or first_word == 'RT':
                return True
        return False

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
def f1_score_single(y_true, y_pred):
    """
    F1_score_single function
    
    This is a performance scoring I created to deal with the multi-label and multi-class problems.
    
    It can be used as scorer for GridSearchCV:
        scorer = make_scorer(multioutput_fscore,beta=1)
        
    Parameters:
        y_true -> labels
        y_prod -> predictions
    
    Output:
        2 * p * r / (p + r)-> customized f1 score for one single data
    """      
    y_true = set(y_true)
    y_pred = set(y_pred)
    cross_size = len(y_true & y_pred)
    if cross_size == 0: return 0.
    p = 1. * cross_size / len(y_pred)
    r = 1. * cross_size / len(y_true)
    return 2 * p * r / (p + r)
    
def f1_score(y_true, y_pred):
    """
    F1_score function  
    It is a sort of arithmetic mean of all the f1_score, computed on each label.

    Parameters:
        y_true -> labels
        y_prod -> predictions

    Output:
        np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)]) -> arithmetic mean of all the f1_score
    """       
    return np.mean([f1_score_single(x, y) for x, y in zip(y_true, y_pred)])

def tokenize(text):
    """
    Tokenize function
    This function is to process your text data
    Parameters:
        text -> list of text messages
    Output:
        clean_tokens -> tokenized text: separated words without upper case
    """     
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    return clean_tokens

# load data
engine = create_engine('sqlite:///../data/disease.db')
df = pd.read_sql_table('df', engine)

# load model
model = joblib.load("../models/classifier.pkl")

# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    # make a distribution plot on the categories data     
    category_names = df.iloc[:,4:].columns
    category_counts = (df.iloc[:,4:]).sum().values
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
    # plot for the distribution of categories
        {
            'data': [
                Bar(
                    x=category_names,
                    y=category_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Categories',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        }
    ]
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()
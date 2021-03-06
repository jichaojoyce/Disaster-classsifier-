"""
Documentation for train_classifier.py
This is a Disaster Response Pipeline Project, Udacity - Data Science Nanodegree
Sample Script Execution:
> python train_classifier.py ../data/DisasterResponse.db classifier.pkl
Inputs:
    1) SQLite db path (containing pre-processed data)
Output:
    2) pickle file name to save ML model
"""
import sys
import numpy as np
import pandas as pd
import re
from sqlalchemy import create_engine
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline,FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import fbeta_score, make_scorer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier,AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.multioutput import MultiOutputClassifier
import pickle
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.neighbors import KNeighborsClassifier
from scipy.stats.mstats import gmean
import joblib
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')

def load_data(database_filepath):
    """
    Load Data function
    This function is to load data from database
    Parameters:
        database_filepath -> path to SQLite db
    Output:
        X -> feature DataFrame named 'message'
        Y -> label DataFrame
        category_names -> used for data visualization (app)
    """    
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('df', engine)
    X = df['message']
    Y = df.iloc[:,4:]    
    category_names = Y.columns
    return X, Y, category_names

def tokenize(text):
    """
    Tokenize function
    This function is to process your text data
    Parameters:
        text -> list of text messages
    Output:
        clean_tokens -> tokenized text: separated words without upper case
    """     
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    tokens = nltk.word_tokenize(text)
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

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

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.starting_verb)
        return pd.DataFrame(X_tagged)
    
def build_model():    
    """
    Build Model function
    
    This function output is a machine learning pipeline. 
    This machine learning pipeline takes in the message column as input and output classification results on the other 36 categories in the dataset. 
    A grid search is used to find better parameters.
    Parameters:
        y_true -> labels
        y_pred -> predictions
    Output:
        cv -> a trained ML pipeline            
    """
    
    pipeline = Pipeline([
        ('features', FeatureUnion([

            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),

            ('starting_verb', StartingVerbExtractor())
        ])),

        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1, 1), (1, 2)),
        'features__text_pipeline__vect__max_df': (0.5,0.75),
        'features__text_pipeline__vect__max_features': (None, 1000),
        'features__text_pipeline__tfidf__use_idf': (True, False)
    }
    scorer = make_scorer(f1_score,greater_is_better = True)
    cv = GridSearchCV(pipeline, parameters,scoring = scorer,verbose=2,n_jobs=-1)
    return cv

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

def evaluate_model(model, X_test, Y_test, category_names):    
    """
    Evaluate Model function
    
    This function is to evaluate the accuracy, precision and recall of the tuned model.
    
    Parameters:
        model -> Scikit ML Pipeline
        X_test -> test features
        Y_test -> test labels
        category_names -> label names (multi-output)
    Outputs:
        Printed average overall accuracy and F1 score.
    """
    
    y_pred_test = model.predict(X_test)
    scorer = f1_score(Y_test,y_pred_test)
    overall_accuracy = (y_pred_test == Y_test).mean().mean()
    print('Average overall accuracy {0:.2f}% \n'.format(overall_accuracy*100))
    print('F1 score (custom definition) {0:.2f}%\n'.format(scorer*100))

def save_model(model, model_filepath):
    """
    Save Model function
    
    This function is to export the trained model as a pickle file.
    
    Parameters:
        model -> Scikit ML Pipeline
        model_filepath -> destination path to save .pkl file
    Outputs:
        A tranined model as a pickle file.
    """    
    filename = model_filepath
    pickle.dump(model, open(filename, 'wb'))

def main():
    """
    Train Classifier Main function
    
    This function applies the Machine Learning Pipeline:
        1) Extract data from SQLite db
        2) Train ML model on training set
        3) Estimate model performance on test set
        4) Save trained model as Pickle
    
    """    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
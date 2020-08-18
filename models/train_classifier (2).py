
import pandas as pd
import sys
import os
import re
from sqlalchemy import create_engine
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
import warnings
import re
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.metrics import classification_report

nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
warnings.simplefilter('ignore')
   
'''' Function loads the disaster Response database and extracts messages and category values from the  database
        The data is split into train- test values.'''
    
def load_data(database_filepath):
    #database_filepath = '../data/DisasterResponse.db'
    #name = 'sqlite:///' + database_filepath
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('DisasterResponse', con=engine)
    df = df[df.related!=2]
    X =  df.message.values
    Y =  df.iloc[:,5:]
    category_names = list(np.array(Y.columns))
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)    
    return X, Y, category_names
    print(df)
    ''' Function used to tokenize lemmatize and lower case the messages values so that it can be passed into our pipeline'''
    
def tokenize(text):
   
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
   
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
  
    return clean_tokens
    '''Eval metrics uses classification report function to print accuracy precision recall and f1 values on the test dataset'''
    
def eval_metrics(array1, array2, col_names):
    metrics = []
    # Evaluate metrics for each set of labels
    for i in range(len(col_names)):
        print(col_names[i])
        print(classification_report(array1[:,i],array2[:,i]))
'''Contains all the preprocessing steps, FeaureExtraction,CountVectorization , creates a pipeline using tokenize Tfidf and randomForestClassifier , parameters are chosen and then are passed into Gridsearch Model'''
    
def build_model(X_train, Y_train):
    pipe = Pipeline([
    ('vect', CountVectorizer(tokenizer = tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf', MultiOutputClassifier(RandomForestClassifier()))
])
    pipe.get_params()

    parameters = {'clf__estimator__max_depth': [10, 50, None],
              'clf__estimator__min_samples_leaf':[2, 5, 10]}

    cv = GridSearchCV(pipe, param_grid = parameters,verbose = 1 , n_jobs =-1)
    cv.fit(X_train, Y_train)
    return cv
''' Prints the values of preicision recall accuracy and f1 score on the test data ''' 

def evaluate_model(model, X_test, Y_test, category_names):
    
    y_predtest = model.predict(X_test)
   # y_predtrain = model.predict(X_train)

    eval_ypred_test = eval_metrics(np.array(Y_test), y_predtest, category_names)
   # eval_ypred_train = eval_metrics(np.array(Y_test), y_predtrain, col_names)
    print(eval_ypred_test)
   # print(eval_ypred_train )

''' Saves the model as a pickle file '''

def save_model(model, model_filepath):
    
    file = pickle.dump(model, open(model_filepath, 'wb'))
''' file runs through main , contains all of the above functions '''
def main():
    if len(sys.argv) == 3:
       database_filepath, model_filepath = sys.argv[1:]    
       print('Loading data...\n    DATABASE: {}'.format(database_filepath))
       X, Y, category_names = load_data(database_filepath)
       X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
       #category_names = list(np.array(Y.columns))
       print('Building model...')
       model = build_model(X_train, Y_train)
      
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
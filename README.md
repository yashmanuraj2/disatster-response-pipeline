                                                          UDACITY DISASTER RESPoNSE PIPELINES : 

The Project contains two datasets , categories and messages , The project classifies messages on the basis of keywords and assigns them to the respective category.
THe first part of the project contains Cleaning of data using NLP pipelines. ETLpreparation
The messages are tokenized , lematized using tokenize text function. The CAtegories columns are extracted from the data and new columns are assigned on the basis of respective  categories.
The new Data set is formed by merging categories data set with messages 
The libraries used  in the ETL pipeline are 
1. pandas
2.sql alchemy engine


The second part of the project contains creating a machine learning model that classifies the messages with thier respective categories
pipeline is used so the CountVectorization, FeatureExtraction and Tfidf can be done parallely.
GridSeacrh CV is used so that best parameters can be choosen to train the dataset.
Decision tree classifier and random forest classifier are used in training the model in which random forest classifier gave better values of recall precision, accuracy
and f1 score and was included in train_Classifier.py
ML-prepration File contains both the models but classification is done on the basis of Random forest classifier. 
The libraries included in ML pipeline are :


sqlalchemy import create_engine
 numpy as np
 sklearn.pipeline import Pipeline
 sklearn.feature_extraction.text import CountVectorizer, TfidfTransformersklearn.model_selection import train_test_split, GridSearchCV
 sklearn.ensemble import RandomForestClassif
 sklearn.pipeline import FeatureUnion
 sklearn.multioutput import MultiOutputClassifier
 sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, make_scorer
 sklearn.model_selection import GridSearchCV
 sklearn.tree import DecisionTreeClassifier
import warnings
import re
import pickle
import nltk
 nltk.corpus import stopwords
nltk.tokenize import word_tokenize
nltk.stem.porter import PorterStemmer
 nltk.stem.wordnet import WordNetLemmatizer
sklearn.metrics import classification_report


                                         RUNNING THE APP :
                                         
           1 . Clone the repository using  git clone https://github.com/yashmanuraj2/disatster-response-pipeline.git
           2. In cd /data run process_data.py
           3. In cd/ models run train_classifier.py
           4. IN app run app.py
           5. type env | grep WORK in new terminal and add ID - domain as the domain name
             https://view6914b2f4-3001.udacity-student-workspaces.com
           id - view6914b2f4-3001
           domain - udacity-student-workspaces.com/ 
           App is running on port 3001
           
           

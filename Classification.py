# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 08:24:12 2018

@author: Faishal
"""

import pandas as pd
import numpy as np
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
from Sastrawi.StopWordRemover.StopWordRemoverFactory import StopWordRemoverFactory
import sklearn.svm
from sklearn import metrics
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split

#input to dataframe
outfile = "D:\\File\\"
dataset = pd.DataFrame.from_csv(outfile + 'ds_asg_data.csv')

#delete row NA
dataset = dataset.dropna()
dataset.isnull().sum()

#List of topic
topic = dataset.groupby('article_topic').size()

#Split text and target
sentence = dataset['article_content'] #text
y = dataset['article_topic'] #target

#preprocessing by regex
sentence = sentence.str.lower()
sentence = sentence.str.replace(r"[^a-zA-Z0-9]+"," ")
sentence = sentence.str.replace(r"([^\w])"," ") 
sentence = sentence.str.replace(r"\b\d+\b", " ")
sentence = sentence.str.replace(r"\s+|\r|\n", " ")
sentence = sentence.str.replace(r"^\s+|\s$", "")

#dataset['article_content'] = sentence

#stemming
factory = StemmerFactory()
stemmer = factory.create_stemmer()

#stopword
factory = StopWordRemoverFactory()
stopword = factory.create_stop_word_remover()

# stemming and stopword process
X = []
index = 1

for item in sentence:
    print('data nomor: {}'.format(index))
    item = stemmer.stem(item)
    item = stopword.remove(item)

    X.append(item)
    index = index + 1
    
X = pd.Series(X)

#if not using stemming and stopword
X = sentence #text

#split to train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 1)

#count vectorizer
count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(X_train)

#tfidf
tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
X_train_tfidf.shape

#SVM
clf = sklearn.svm.LinearSVC()

#make pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                      ('tfidf', TfidfTransformer()),
                      ('clf', sklearn.svm.LinearSVC()),
 ])

#fitting model
text_clf = text_clf.fit(X_train, y_train)

#predict
predicted = text_clf.predict(X_test)

#score validation
final_score = np.mean(predicted == y_test)
print("Accuracy: ",final_score*100,"%")

#Classification Report
classification_report = metrics.classification_report(y_test, predicted, target_names=y_test)

#Confusion Matrix
confusion_matrix = metrics.confusion_matrix(y_test, predicted)

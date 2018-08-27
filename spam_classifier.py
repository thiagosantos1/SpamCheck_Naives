#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 26 16:43:44 2018

@author: thiago
"""

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.cross_validation import train_test_split
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import string
import nltk
from sklearn.metrics import confusion_matrix
#nltk.download('punkt')

# Pre_process the data
def data_preprocess(dataset_file):
    # Read the dataset
    dataset = pd.read_csv(dataset_file, encoding = "ISO-8859-1")
    X = dataset.iloc[:, [1]].values
    y = dataset.iloc[:, 0].values

    # convert to np array
    X = np.array(X, dtype = 'str')
    y = np.array(y, dtype = 'str')

    # encode the label.
    # 0 means ham
    # 1 means spam
    LabelEncoder_X = LabelEncoder()
    y = LabelEncoder_X.fit_transform(y)


    return X,y


def make_bag_of_words(dataset):
    bag_of_words = []
    table = str.maketrans('', '', string.punctuation)
    stop_words = set(stopwords.words('english'))
    for email in dataset:
        for line in email:
            # Clean data

            # remove ponctuation
            words = word_tokenize(line)
            words = [word.lower() for word in words ]

            # remove punctuation from each word
            words = [w.translate(table) for w in words]

            # remove remaining tokens that are not alphabetic
            words = [word for word in words if word.isalpha()]

            # Filter out Stop Words (and Pipeline). Remove non inportant words
            words = [w for w in words if not w in stop_words]

            # Steam words --> Remove ing for example
            porter = PorterStemmer()
            words = [porter.stem(word) for word in words]

            # add to the bag of words
            bag_of_words += words

           # remove duplicates and return bag of words
    return list(set(bag_of_words))

# We can build our feature matrix, which will be based on
# how many times each word appears in an email
# our classier will use this matrix to calculate the distribution probability, and find the correlation
def build_features(dataset, bag_of_words):
    # nparray to have the features
    features_matrix = np.zeros((len(dataset), len(bag_of_words)))

    # collecting the number of occurances of each of the words in the emails
    for email_index,email in enumerate(dataset):
        for line in email:
            words = line.split()
            for word_index, word in enumerate(bag_of_words):
                features_matrix[email_index, word_index] = words.count(word)

    return features_matrix


X,y = data_preprocess('dataset/spam.csv')
bag_of_words = make_bag_of_words(X)

# split data in training and test set
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2)

# classifier, using Naive Bayes Multinomial
classifier = MultinomialNB()

# create matrix of features, for training and test
features_train = build_features(X_train, bag_of_words)
features_test = build_features(X_test, bag_of_words)

# train our ML
classifier.fit(features_train, y_train)

# predict
y_pred = classifier.predict(features_test)  # vector of predictions


# Making the Confusion Matrix - Evaluate our model. Check for how many correct predictions
cm = confusion_matrix(y_test,y_pred)

accuracy = classifier.score(features_test, y_test)







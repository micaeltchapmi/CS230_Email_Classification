import os
import json
import string
#import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pandas as pd
import glob
import numpy as np
from sklearn.metrics import classification_report,confusion_matrix
# Get test, train and val data
save_dir = "./processed_data"
def main():
    files_train = glob.glob(save_dir + "/Train/*/*/*")
    files_test = glob.glob(save_dir + "/Test/*/*/*")
    
    # Saving TFID vectors and labels
    train_vectors =[]
    train_text =[]
    test_text =[]
    train_labels =[]
    test_vectors =[]
    test_labels =[]
    for i in files_train:
        vector_i = json.load(open(i))
        train_vectors.append(vector_i["Text_Vector"])
        train_text.append(vector_i["Text"])
        train_labels.append(vector_i["Label"])
    
    
    for i in files_test:
        vector_i = json.load(open(i))
        test_vectors.append(vector_i["Text_Vector"])
        test_text.append(vector_i["Text"])
        test_labels.append(vector_i["Label"])
    print(len(test_labels))
    #CountVectorizer
    vectorizer=TfidfVectorizer(max_features=1000)
    spamham_countVectorizer_train=vectorizer.fit_transform(train_text)
    spamham_countVectorizer_test=vectorizer.transform(test_text)

    # Training
    NB_classifier=MultinomialNB()
    NB_classifier.fit(spamham_countVectorizer_train, train_labels)
    y_predict_train=NB_classifier.predict(spamham_countVectorizer_train)
    y_predict_train

    #Testing
    y_predict_test=NB_classifier.predict(spamham_countVectorizer_test)
    y_predict_test

    print(classification_report(test_labels,y_predict_test))

if __name__ == "__main__":
    main()

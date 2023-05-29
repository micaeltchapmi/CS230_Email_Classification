#Source: https://www.kaggle.com/code/jyotishmandas/spam-text-classification-using-decision-tree/notebook
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import re
import glob
import json
np.random.seed(1)

data_dir = "./processed_data"
files = glob.glob(data_dir + "/*/*/*/*")
np.random.shuffle(files)
#load all emails and convert to pandas table

emails = []
for f in files:
    emails.append(json.load(open(f)))
df = pd.DataFrame(emails)

#Get training and test data
train_df = df[df["set"] == "Train"]
test_df = df[df["set"] == "Test"]

train_labels = train_df["Label"].astype(np.int32)
train_text_vector = train_df["Text_Vector"].tolist()
# Convert lists in DataFrame to arrays

test_labels = test_df["Label"].astype(np.int32)
test_text_vector = test_df["Text_Vector"].tolist()

def prediction(X_test, model_object):
    # Predicton on test with giniIndex
    y_pred = model_object.predict(X_test)
    #print("Predicted values:")
    #print(y_pred)
    return y_pred

def cal_accuracy(y_test, y_pred):
      
    #print("Confusion Matrix: ",
    #    confusion_matrix(y_test, y_pred))
      
    print ("Accuracy : ",
    accuracy_score(y_test,y_pred)*100)

def main():

    # Decision tree with gini
    print("Evaluating decision trees with gini criterion...")
    model_gini = DecisionTreeClassifier(criterion = "gini",
                random_state = 123,max_depth=10, min_samples_leaf=6)
    
    # Performing training
    model_gini.fit(train_text_vector,train_labels)

    # Prediction using gini
    print("Train accuracy...")
    y_pred_gini = prediction(train_text_vector, model_gini)
    cal_accuracy(train_labels, y_pred_gini)
    print("Test accuracy...")
    y_pred_gini = prediction(test_text_vector, model_gini)
    cal_accuracy(test_labels, y_pred_gini)

    print("\nEvaluating decision trees with entropy criterion...")
    # Decision tree with entropy
    model_entropy = DecisionTreeClassifier(
                criterion = "entropy", random_state = 123,
                max_depth = 10, min_samples_leaf = 6)
    
    # Performing training
    model_entropy.fit(train_text_vector, train_labels)

    # Prediction using entropy
    print("Train accuracy...")
    y_pred_entropy = prediction(train_text_vector, model_entropy)
    cal_accuracy(train_labels, y_pred_entropy)
    print("Test accuracy...")
    y_pred_entropy = prediction(test_text_vector, model_entropy)
    cal_accuracy(test_labels, y_pred_entropy)


if __name__=="__main__":
    main()
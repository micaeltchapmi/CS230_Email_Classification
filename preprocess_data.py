import os
import json
import string
import nltk
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
import pandas as pd
import glob
import numpy as np

save_dir = "./processed_data"
Email_Categories = {"0": "Passed_Events",
                    "1": "Fufilled_requests",
                    "2": "Automated_Responses"}
Labels = {"0": "Delete",
          "1": "Keep"}


def save_label(Email_Data):
    # Email_Data: dictionary containing email content and metadata information for a single email
    # label: 1: keep, 0: delete
    # Category: Number indicating one of the predefined Email Categories
    Category = Email_Categories[Email_Data["Category"]]
    Label = Labels[Email_Data["Label"]]

    ID = Email_Data["email_id"]
    Set = Email_Data["set"]

    out_dir = os.path.join(save_dir, Set, Label, Category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ID + ".json")

    with open(out_path, 'w') as f:
        json.dump(Email_Data, f)


def main():
    data_dir = "./data"
    files = glob.glob(data_dir + "/*/*/*")

    # DO THE SPLIT TRAIN/VAL/TEST HERE AND RUN
    Train, Val, Test = [], [], []

    for l in os.listdir(data_dir):
        for c in os.listdir(os.path.join(data_dir, l)):
            files_c = os.listdir(os.path.join(data_dir, l, c))
            n = len(files_c)
            np.random.shuffle(files_c)
            train, val, test = files_c[0:int(0.7 * n)], files_c[int(0.7 * n): int(0.9 * n)], files_c[int(0.9 * n):]
            train = [os.path.join(data_dir, l, c, k) for k in train]
            val = [os.path.join(data_dir, l, c, k) for k in val]
            test = [os.path.join(data_dir, l, c, k) for k in test]
            Train.extend(train)
            Val.extend(val)
            Test.extend(test)

    # Create dataframe and add field to remember which files were train/val
    emails = []
    for f in Train:
        email_i = json.load(open(f))
        email_i["set"] = "Train"
        emails.append(email_i)
    for f in Val:
        email_i = json.load(open(f))
        email_i["set"] = "Val"
        emails.append(email_i)
    for f in Test:
        email_i = json.load(open(f))
        email_i["set"] = "Test"
        emails.append(email_i)

    df = pd.DataFrame.from_dict(emails)
    # Preprocess the train and validation

    # Data cleaning
    print(df)
    df['Text'] = df['Text'].apply(lambda x: x.lower())  # Convert all text to lowercase
    df['Text'] = df['Text'].apply(
        lambda x: x.translate(str.maketrans('', '', string.punctuation)))  # Remove punctuation marks
    stop_words = set(stopwords.words('english'))

    df['Text'] = df['Text'].apply(
        lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))  # Remove stop words

    # Lemmatization 

    df['Text'] = df['Text'].apply(
        lambda x: ' '.join([word for word in word_tokenize(x) if word not in stop_words]))  # Remove stop words

    # Vectorization

    # vectorizer = CountVectorizer() #counts number of times word appears in text
    vectorizer = TfidfVectorizer()  # better because uses both count and importance of words in corpus

    # Fit vectorizer only on train then transform on Val and test data to avoid data leakage
    train_pos = df["set"].isin(["Train"])
    vectorizer.fit(df[train_pos]['Text'])

    # Transform on train val and test
    Text_Vectors = vectorizer.transform(df['Text'])

    # add text vectors to dataframe
    df["Text_Vector"] = Text_Vectors.toarray().tolist()

    # Split data
    email_data = df.to_dict(orient="records")
    for e in email_data:
        save_label(e)


if __name__ == "__main__":
    main()

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
data_dir = "./data"
Email_Categories = {"0": "Important",
                    "1": "Social",
                    "2": "Promotions",
                    "3": "Other_Unimportant",
                    }
Labels = {"0": "Delete",
          "1": "Keep"}
max_n_features = 1000 # For TFID vectorizer. Might need tuning


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
    files = glob.glob(data_dir + "/*/*/*")

    # DO THE SPLIT TRAIN/VAL/TEST HERE AND RUN
    delete_files = glob.glob(data_dir + "/Delete/*/*")
    keep_files = glob.glob(data_dir + "/Keep/*/*") 

    #for balancing the data
    n = 250
    delete_files = delete_files[0:n]
    keep_files = keep_files[0:n]
    
    np.random.shuffle(delete_files)
    np.random.shuffle(keep_files)

    #Downsample class label with more emails to balance the dataset
    if len(delete_files) > len(keep_files):
        delete_files = delete_files[0:len(keep_files)]
    else:
        keep_files = keep_files[0:len(delete_files)]
    
    #make sure the dataset is balanced between keep and delete . Will raise error if not the case
    assert len(delete_files) == len(keep_files)

    #split emails from each category into 70/20/10 and combine into a train/val/test split
    n = len(keep_files) # same as len(delete files)
    Train_k, Val_k, Test_k = keep_files[0:int(0.7 * n)], keep_files[int(0.7 * n): int(0.9 * n)], keep_files[int(0.9 * n):]
    Train_d, Val_d, Test_d = delete_files[0:int(0.7 * n)], delete_files[int(0.7 * n): int(0.9 * n)], delete_files[int(0.9 * n):]
    Train, Val, Test = Train_k + Train_d , Val_k + Val_d , Test_k + Test_d

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
    vectorizer = TfidfVectorizer(max_features=max_n_features)  # better because uses both count and importance of words in corpus

    # Fit vectorizer only on train then transform on Val and test data to avoid data leakage
    train_pos = df["set"].isin(["Train"])
    vectorizer.fit(df[train_pos]['Text'])

    """
    #get features and their frequencies to determe threshold. This helps manually set max_n_features above 

    # Get the vocabulary and feature indices
    vocabulary = vectorizer.vocabulary_
    feature_indices = vectorizer.get_feature_names()
    # Count the frequency of each word in the corpus
    word_frequency = {}
    for word, index in vocabulary.items():
        word_frequency[word] = sum([1 for doc in df[train_pos]['Text'] if feature_indices[index] in doc])
    # Print the frequency of each word in the vocabulary
    for word, frequency in word_frequency.items():
        print(f"{word}: {frequency}")
    """

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

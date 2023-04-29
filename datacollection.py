""""
Data collection
    1. Read email
    2. Label (keep or delete)
    3. Label (Category e.g 1. Category1, 2. Category 2, etc)
    4. Saving
        a) Folder Structure
           Category1
           Keep
           Delete
        b) Saved Email Format
           {“timestamp”: time, “Content”: text, “Title:” title, etc}

"""

import os
import json
import numpy as np

Email_Categories = {"0" : "Passed Events",
                    "1" : "Fufilled requests",
                    "2" : "Automated Responses"}
Labels = {"0": "Delete",
          "1": "Keep"}

save_dir = "./data"

#TODO
#1. save labels
#2. Read and label emails
#   a. label as keep (1) or delete(0)
#   b. label category
#   c. skip emails that have already been labeled and saved already
#   d. Skip emails that may be too subjective to reduce bias
#   e. Add new categories you discover to the Email_Categories list

def save_label(Email_Data):
    #Email_Data: dictionary containing email content and metadata information for a single email
    #label: 1: keep, 0: delete
    #Category: Number indicating one of the predefined Email Categories
    Category = Email_Categories[Email_Data["Category"]]
    Label = Labels[Email_Data["Label"]]
    ID =  Email_Data["email_id"]

    out_dir = os.path.join(save_dir, Label, Category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ID + ".json")
    
    with open(out_path, 'w') as f:
        json.dump(Email_Data, f)


def test_save_label():
    data1 = {"email_id": "00001" ,
            "Category": "1" ,
            "Label": "1",
            "Text": "Your request has been fulfilled",
            "Sender": "Paul@stanford.edu"
            }
    #TODO: add other fields that could be useful like:
    #Number of replies, attachments, etc
    data2 = {"email_id": "00002" ,
            "Category": "2" ,
            "Label": "0",
            "Text": "Do not reply. This is an automated response",
            "Sender": "bot@amazon.org"
            }
    save_label(data1)
    save_label(data2)


def main():
    test_save_label()

if __name__ == "__main__":
    main()
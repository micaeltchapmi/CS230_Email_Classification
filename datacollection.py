import os
import json
import numpy as np
import imaplib
import email
import nltk
from nltk.tokenize import word_tokenize
from email.header import decode_header
import glob
from imap_tools import MailBox, AND

# Server is the address of the IMAP server


from sympy import categories

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

Email_Categories = {"0": "Passed_Events",
                    "1": "Fufilled_requests",
                    "2": "Automated_Responses"}

Labels = {"0": "Delete",
          "1": "Keep"}

save_dir = "./data"


# TODO
# 1. save labels
# 2. Read and label emails
#   a. label as keep (1) or delete(0)
#   b. label category
#   c. skip emails that have already been labeled and saved already
#   d. Skip emails that may be too subjective to reduce bias
#   e. Add new categories you discover to the Email_Categories list

def save_label(Email_Data):
    # Email_Data: dictionary containing email content and metadata information for a single email
    # label: 1: keep, 0: delete
    # Category: Number indicating one of the predefined Email Categories
    Category = Email_Categories[Email_Data["Category"]]
    Label = Labels[Email_Data["Label"]]
    ID = Email_Data["email_id"]

    out_dir = os.path.join(save_dir, Label, Category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ID + ".json")

    with open(out_path, 'w') as f:
        json.dump(Email_Data, f)


def test_save_label():
    # TODO: add other fields that could be useful like:
    # Number of replies, attachments, etc

    for i, k in enumerate(range(36)):
        id = "%05d" % (i)
        Category = str((i % 3))
        Label = str((i % 2))
        Text = "%d message" % (i)
        Sender = "%i@domain.edu" % (i)
        data = {"email_id": id,
                "Category": Category,
                "Label": Label,
                "Text": Text,
                "Sender": Sender
                }
        save_label(data)


# account credentials
def read_email():
    username = "brianlangat11@gmail.com"
    password = ""

    """"
    # create an IMAP4 class with SSL
    imap = imaplib.IMAP4_SSL("imap.gmail.com")
    # authenticate
    imap.login(username, password)

    # select the mailbox I want to delete in
    # if you want SPAM, use imap.select("SPAM") instead
    imap.select("INBOX")
    # to get all mails
    status, messages = imap.search(None, 'FROM "faridacheptoo@gmail.com"')
    # print(messages)

    for num in messages[0].split():
        result, data = imap.fetch(num, '(RFC822)')
        msg = data[0][1]
       # print(data[0][1])

        print('Subject:', msg['Subject'])
        # print('From:', msg['From'])
        break
        """
    mb = MailBox('imap.gmail.com').login(username, password)

    # Fetch all unseen emails containing "xyz.com" in the from field
    # Don't mark them as seen
    # Set bulk=True to read them all into memory in one fetch
    # (as opposed to in streaming which is slower but uses less memory)
    messages = mb.fetch(criteria=AND(seen=True, from_="faridacheptoo@gmail.com"),
                        mark_seen=True,
                        bulk=True)

    # Adding emails into a dictionary
    saved_data = {}
    for msg in messages:
        files = glob.glob(save_dir + "/*/*/*")

        # For loop to check if the data is already labelled
        for file in files:
            info = file.split("\\")
            info2 = info[-1]
            info3 = info2.split(".")
            myinfo = info3[0]
            saved_data[myinfo] = myinfo

    for msg in messages:
        if msg.uid in saved_data:
            continue

        category_label, class_label = label_category(
            msg)  # Call the function label_category for the user to input the labels in the training data
        if class_label == -1:
            continue
        Email_Data = {"Category": category_label,
                      "Label": class_label,
                      "email_id": msg.uid,
                      "Text": msg.txt}

        save_label(Email_Data)  # Calling save_label with email data


# Define a function to label an email based on its content
def label_category(email):
    """Prompts the user to label an email"""
    prompt1 = "Input the category \n 0: Passed_Events, 1: Fufilled_requests, 2: Automated_Responses"
    prompt2 = "Input the labels \n 0: Delete, 1: Keep  -1: Skip"

    print(email.from_, ': ', email.subject)
    print(email.text)
    label1 = input(prompt1)
    label2 = input(prompt2)
    return label1, label2


def main():
    read_email()


if __name__ == '__main__':
    main()

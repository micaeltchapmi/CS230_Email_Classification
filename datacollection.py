import os
import json
from datetime import datetime
import numpy as np
import imaplib
import email
from email.header import decode_header
import glob
import base64
import html
import re

from google_auth_oauthlib.flow import InstalledAppFlow
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request
from googleapiclient.discovery import build
from bs4 import BeautifulSoup

#TODO:
# Add ability to relabel some threads if to help correct some of the labels based on our findings if needed
# Add other email filters like date, time, etc to make labeling easier and more trackable

Email_Categories = {"0": "Passed_Events",
                    "1": "Fufilled_requests",
                    "2": "Automated_Responses",
                    "3": "Important"
                    }

Labels = {"0": "Delete",
          "1": "Keep"}

save_dir = "./data"

### Gmail Mailbox credentials ###
username = "user@gmail.com"
password = ""

#### Gmail API settings ###

credentials_dir = "./credentials"
path_to_creds = os.path.join(credentials_dir, "gmail_api_creds.json")
path_to_refresh_token = os.path.join(credentials_dir, "refresh_token.json")

# If modifying these SCOPES, delete the file token.json and generate new one.
SCOPES = ['https://www.googleapis.com/auth/gmail.readonly']

# Function to save a labeled email
def save_label(Email_Data):
    # Email_Data: dictionary containing email content and metadata information for a single email
    # label: 1: keep, 0: delete
    # Category: Number indicating one of the predefined Email Categories
    Category = Email_Categories[Email_Data["Category"]]
    Label = Labels[Email_Data["Label"]]
    ID = Email_Data["thread_id"] + "_" + Email_Data["email_id"]

    out_dir = os.path.join(save_dir, Label, Category)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, ID + ".json")

    with open(out_path, 'w') as f:
        json.dump(Email_Data, f)

################################################################################################################
####                    Helper Functions for Reading and Labeling Threads                                   ####
################################################################################################################

### Function to Read gmail API credentials and get Refresh Token ###
def get_credentials():
    creds = None
    # The file token.json stores the user's access and refresh tokens, and is
    # created automatically when the authorization flow completes for the first
    # time.
    if os.path.exists(path_to_refresh_token):
        creds = Credentials.from_authorized_user_file(path_to_refresh_token, SCOPES)

    # If there are no (valid) credentials available, let the user log in.
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_config(json.loads(open(path_to_creds, 'r').read()), SCOPES)
            creds = flow.run_local_server(port=0)
        # Save the credentials for the next run
        with open(path_to_refresh_token, 'w') as token:
            token.write(creds.to_json())

    return creds

#Function to read all threads from most recent
def Get_All_Threads():
    # Get credentials
    credentials = get_credentials()

    # Build the Gmail API client
    service = build('gmail', 'v1', credentials=credentials)

    # Filter threads
    LABEL_FILTER = "in:inbox category:primary"
    USERID="me"
    

    # Call the Gmail API to get the user's primary inbox messages
    #TODO add more filters e.g date, time of day, month, year, 
    response = service.users().messages().list(
        userId=USERID,
        q=LABEL_FILTER,
        maxResults=10
    ).execute()
    messages = response.get('messages', [])

    threads = []
    if not messages:
        print('No messages found. Check your search criteria')
        return
    else:
        thread_ids = []
        for message in messages:
            thread_id = message['threadId']
            if thread_id not in thread_ids:
                thread_ids.append(thread_id)

        # Retrieve threads by ID and sort them in the order they are displayed in the user's Gmail account
        
        for thread_id in thread_ids:
            response = service.users().threads().get(
                userId=USERID,
                id=thread_id,
                format='full'
            ).execute()
            messages = response['messages']
            threads.append({'id': thread_id, 'messages': messages})

        threads = sorted(threads, key=lambda t: t['messages'][-1]['internalDate'], reverse=True)
    
    return threads

# Function to get text content from body of email 
def get_message_body(message):
    text_parts = []
    if "parts" in message.keys():
        message_parts = message['parts']
        for part in message_parts:
            if part['mimeType'] == 'text/plain':
                data = part['body'].get('data')
                if data:
                    text = base64.urlsafe_b64decode(data).decode('utf-8')
                    text = re.sub(r'\[cid:.*?\]', '', text) # remove CID URLs
                    text_parts.append(text)

            if part.get('parts'):
                text_parts += get_message_body(part)

    elif "body" in message.keys():
        data = message['body'].get('data')
        if data:
            text = base64.urlsafe_b64decode(data).decode('utf-8')
            soup = BeautifulSoup(text, 'html.parser')
            decoded_html = html.unescape(soup.get_text())
            text = re.sub(r'\[cid:.*?\]', '', decoded_html) # remove CID URLs
            text_parts.append(text)

    return ''.join(html.unescape(text_parts))

# Function to get header info from email
def get_header_info(message):
    subject, sender = "", ""
    headers = message['headers']
    for header in headers:
        if header['name'] == 'Subject':
            subject = header['value']
        if header['name'] == 'From':
            sender = header['value']
    return subject, sender
    

################################################################################################################
####                    Main code for Reading and Labeling Threads                                          ####
################################################################################################################

# Function to read and label all emails from primary tab
# Email threads are sorted to be displayed in same order as they appear on gmail
# Only the first message for each thread is used for labeling
def Label_Emails():
    #Get previously labeled threads from disk
    prev_labeled_threads = set()
    files = glob.glob(save_dir + "/*/*/*")
    for file in files:
        thread_id = file.split("/")[-1].split(".")[0]
        prev_labeled_threads.add(thread_id)

    # Add previously skipped threads to dictionary
    skipped_threads_file = "./skipped_threads.txt"
    prev_skipped_threads = set()
    skipped_threads = open(skipped_threads_file, 'r').readlines()
    skipped_threads = [k.strip("\n") for k in skipped_threads] #remove newline characters
    for s in skipped_threads:
        prev_skipped_threads.add(s)
    
    # Get new threads to label
    threads = Get_All_Threads()

    # Here is the easy part: We now Label the threads
    for thread in threads:
        #get ids
        first_message = thread['messages'][0]
        thread_id = first_message["threadId"]

        #skip if thread already labeled or previously skipped
        if (thread_id in prev_labeled_threads) or (thread_id in prev_skipped_threads):
            continue

        message_id = first_message["id"]
        body = get_message_body(first_message["payload"])
        subject, sender = get_header_info(first_message["payload"])

        #TODO Later: Get other email info if needed

        # label current thread
        Class, Category = Label_Thread(sender, subject, body)

        #skip emails that were too subjective and add skipped IDs to file
        if Class == "-1":
            prev_skipped_threads.add(thread_id)
            with open(skipped_threads_file, 'w') as f:
                for t in prev_skipped_threads:
                    f.write("%s\n" % (t))

            continue

        Email_Data = {"Category": Category,
                      "Label": Class,
                      "thread_id": thread_id,
                      "email_id": message_id,
                      "Text": body}
        
        #save label
        save_label(Email_Data)


# Function to label a thread
def Label_Thread(sender, subject, body):
    """Prompts the user to label an email"""
    label_prompt = "Input the label \n 0: Delete, 1: Keep  -1: Skip : "
    cat_prompt = "Input the category \n"
    for i in range(len(Email_Categories)):
        cat_prompt += "%d %s, " % (i, Email_Categories[str(i)])

    print("From: " , sender)
    print("Subject: ", subject)
    #limit body to 300 characters
    print(body[0:300]) 
    Class = input(label_prompt)

    assert Class in ["0", "1", "-1"]

    #no need to label category if we skipped
    if Class == "-1":
        return Class, None
    
    Category = input(cat_prompt)

    assert Category in Email_Categories.keys()

    return Class, Category


def main():
    Label_Emails()


if __name__ == '__main__':
    main()

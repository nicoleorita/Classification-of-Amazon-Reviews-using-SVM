# -*- coding: utf-8 -*-
"""
Created on Mon Mar 26 16:32:18 2018

@author: Nicole Rita
"""

import json
import keras
import nltk
import pandas as pd
import numpy as np
import re
import codecs


path = 'C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One\\Beauty_5.json'
fopen = open(path, 'r')


#Way to open .json given by the teacher
while 1:
    line = fopen.readline()
    sample = json.loads(line)
    print(sample) # Look at the different fields
    reviewText = sample['reviewText']
    fopen.close()


#Opens json in another way
json_data = []
for line in open(path):
    json_data.append(json.loads(line))

json_data.describe()

#select random 100.000 rows
import random

num_to_select = 100000                    # Set the number to select here.
list_of_random_items = random.sample(json_data, num_to_select)

finalOneHundred = list()
for dictionary in list_of_random_items:
        finalOneHundred.append(dictionary.get('reviewText'))
        
    
# ------------------------------

        
 #Convert list to dataframe 
df = pd.DataFrame(finalOneHundred)
#Give name to the only column
df.columns = ['Review Text']
df.head()
df.describe()


#clean pontuation and signs from text
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[^A-Za-z0-9(),!?@\'\`\"\_\n]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

clean_df = standardize_text(df, 'Review Text')

clean_df.to_csv("clean_data.csv")
clean_df.head()
clean_df.tail()

#TOKENIZING
from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer(r'\w+')

clean_df["tokens"] = clean_df["Review Text"].apply(tokenizer.tokenize)
clean_df.head()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

all_words = [word for tokens in clean_df["tokens"] for word in tokens]
sentence_lengths = [len(tokens) for tokens in clean_df["tokens"]]
VOCAB = sorted(list(set(all_words)))
print("%s words total, with a vocabulary size of %s" % (len(all_words), len(VOCAB)))
print("Max sentence length is %s" % max(sentence_lengths))


import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Sentence length')
plt.ylabel('Number of sentences')
plt.hist(sentence_lengths)
plt.show()

#BAG OF WORDS
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

def cv(data):
    count_vectorizer = CountVectorizer()

    emb = count_vectorizer.fit_transform(data)

    return emb, count_vectorizer

list_corpus = clean_df["Review Text"].tolist()
list_labels = clean_df["tokens"].tolist()

X_train, X_test, y_train, y_test = train_test_split(list_corpus, list_labels, test_size=0.2, 
                                                                                random_state=40)

X_train_counts, count_vectorizer = cv(X_train)
X_test_counts = count_vectorizer.transform(X_test)














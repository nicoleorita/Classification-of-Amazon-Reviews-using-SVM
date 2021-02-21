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
from re import compile, IGNORECASE, DOTALL
import codecs


path = 'C:/Users/eduar/Videos/Beauty_5.json'
fopen = open(path, 'r')

'''
# #1 - Way to open .json given by the teacher
while 1:
    line = fopen.readline()
    sample = json.loads(line)
    print(sample) # Look at the different fields
    reviewText = sample['reviewText']
    fopen.close()
'''

#Opens json in another way
json_data = []
for line in open(path):
    json_data.append(json.loads(line))


#select random 100.000 rows
import random

num_to_select = 100000                    # Set the number to select here.
list_of_random_reviews = random.sample(json_data, num_to_select)

finalOneHundred = list()
for dictionary in list_of_random_reviews:
        finalOneHundred.append(dictionary.get('reviewText'))
    

# ------------------------------ CLEAN TEXT -------------------------------------

        
 #Convert list to dataframe 
df = pd.DataFrame(finalOneHundred)
#Give name to the only column
df.columns = ['Review Text']
df.head()
df.tail()
df.describe()

#clean pontuation and signs from text
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[\.\?\!,\$\&\*\%\(\)\~\-\"\'\]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df
clean_df = standardize_text(df, 'Review Text')

#-------------------------------------EXPAND CONTRACTIONS ---------------------------
#--------------DICIONARIO COM CONTRAÇÕES
CONTRACTION_DICT = {
    "ain't": "is not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he'll've": "he he will have",
    "he's": "he is",
    "how'd": "how did",
    "how'd'y": "how do you",
    "how'll": "how will",
    "how's": "how is",
    "I'd": "I would",
    "I'd've": "I would have",
    "I'll": "I will",
    "I'll've": "I will have",
    "I'm": "I am",
    "I've": "I have",
    "i'd": "i would",
    "i'd've": "i would have",
    "i'll": "i will",
    "i'll've": "i will have",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'd've": "it would have",
    "it'll": "it will",
    "it'll've": "it will have",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "mightn't've": "might not have",
    "must've": "must have",
    "mustn't": "must not",
    "mustn't've": "must not have",
    "needn't": "need not",
    "needn't've": "need not have",
    "o'clock": "of the clock",
    "oughtn't": "ought not",
    "oughtn't've": "ought not have",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "shan't've": "shall not have",
    "she'd": "she would",
    "she'd've": "she would have",
    "she'll": "she will",
    "she'll've": "she will have",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "shouldn't've": "should not have",
    "so've": "so have",
    "so's": "so as",
    "that'd": "that would",
    "that'd've": "that would have",
    "that's": "that is",
    "there'd": "there would",
    "there'd've": "there would have",
    "there's": "there is",
    "they'd": "they would",
    "they'd've": "they would have",
    "they'll": "they will",
    "they'll've": "they will have",
    "they're": "they are",
    "they've": "they have",
    "to've": "to have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'd've": "we would have",
    "we'll": "we will",
    "we'll've": "we will have",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what'll've": "what will have",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "when's": "when is",
    "when've": "when have",
    "where'd": "where did",
    "where's": "where is",
    "where've": "where have",
    "who'll": "who will",
    "who'll've": "who will have",
    "who's": "who is",
    "who've": "who have",
    "why's": "why is",
    "why've": "why have",
    "will've": "will have",
    "won't": "will not",
    "won't've": "will not have",
    "would've": "would have",
    "wouldn't": "would not",
    "wouldn't've": "would not have",
    "y'all": "you all",
    "y'all'd": "you all would",
    "y'all'd've": "you all would have",
    "y'all're": "you all are",
    "y'all've": "you all have",
    "you'd": "you would",
    "you'd've": "you would have",
    "you'll": "you will",
    "you'll've": "you will have",
    "you're": "you are",
    "you've": "you have"
}
def expand_contractions(text):
    """Expands contractions in text."""
    # Creates contractions pattern.
    contractions_pattern = compile('({})'.format('|'.join(CONTRACTION_DICT.keys())), flags=IGNORECASE | DOTALL)

    def expand_match(contraction):
        """Expands matched contraction."""
        # Retrieves matched contraction from string.
        match = contraction.group(0)
        # Stores first character for case sensitivity.
        first_char = match[0]
        # Find expanded contraction in dictionary, based on contraction key.
        expanded_contraction = CONTRACTION_DICT.get(match)
        # If the contraction could not be found, try again with lower case.
        if not expanded_contraction:
            expanded_contraction = CONTRACTION_DICT.get(match.lower())
        # Add first character to expanded contraction.
        expanded_contraction = first_char + expanded_contraction[1:]
        return expanded_contraction

    # Replaces contractions with expanded contractions in text.
    expanded_text = contractions_pattern.sub(expand_match, text)
    return expanded_text

clean_text = []
for line in clean_df.get('Review Text'):
    clean_text.append(expand_contractions(line))
    
clean_df.to_csv("clean_data.csv")
clean_df.head()
clean_df.tail()

#-----------------------------------------------TOKENIZING--------------------------------
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




#Graph with sentence lenght vs. nr of sentences
import matplotlib.pyplot as plt

fig = plt.figure(figsize=(10, 10)) 
plt.xlabel('Sentence length')
plt.ylabel('Number of sentences')
plt.hist(sentence_lengths)
plt.show()

#---------------------------------------BAG OF WORDS----------------------------------------
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


from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(C=30.0, class_weight='balanced', solver='newton-cg', 
                         multi_class='multinomial', n_jobs=-1, random_state=40)
clf.fit(X_train_counts, y_train)

y_predicted_counts = clf.predict(X_test_counts)











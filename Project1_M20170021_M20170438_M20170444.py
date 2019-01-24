## Pre-Processing
# The code was runned one time for the Train data and again to the Test data.

import json
import nltk
import pandas as pd
import numpy as np
import re
import random
import sys
sys.path.append("C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One")
from Contractions import CONTRACTION_DICT
from re import compile, IGNORECASE, DOTALL, sub
from nltk.corpus import wordnet

# Open and Read the files

path_beauty = "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One\\Beauty_5.json"
path_grocery = "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One\\Grocery_and_Gourmet_Food_5.json"
path_movies_tv = "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One\\Movies_and_TV_5.json"
path_toys_games = "C:\\Users\\Nicole Rita\\Documents\\NOVA IMS\\2nd Semester\\Text Analytics\\PROJECT\\Project_One\\Toys_and_Games_5.json"

fopen = open(path_beauty, 'r')
fopen = open(path_grocery, 'r')
fopen = open(path_movies_tv, 'r')
fopen = open(path_toys_games, 'r')


#IMPORTING BEAUTY DATASET
json_data = []
for line in open(path_grocery):
    json_data.append(json.loads(line))

# Divide the data based on the Overall rating (make it easier to use in the next project)

overall1 = list()
overall2 = list()
overall3 = list()
overall4 = list()
overall5 = list()

for j in json_data:
    if j.get('overall') == 1.0:
              overall1.append(j)
    elif j.get('overall') == 2.0:
            overall2.append(j)
    elif j.get('overall') == 3.0:
            overall3.append(j)
    elif j.get('overall') == 4.0:
            overall4.append(j)
    elif j.get('overall') == 5.0:
            overall5.append(j)

def calculate_percentage ():

    totalLength = len(json_data)

    percentageL1 = (len(overall1) * 1 ) / totalLength
    percentageL2 = (len(overall2) * 1 ) / totalLength
    percentageL3 = (len(overall3) * 1 ) / totalLength
    percentageL4 = (len(overall4) * 1 ) / totalLength
    percentageL5 = (len(overall5) * 1 ) / totalLength

    num_to_select1 = round(percentageL1 *100000)
    num_to_select2 = round(percentageL2 *100000)
    num_to_select3 = round(percentageL3 *100000)
    num_to_select4 = round(percentageL4 *100000)
    num_to_select5 = round(percentageL5 *100000)

    final_list = list()
    final_list.extend(random.sample(overall1, num_to_select1))
    final_list.extend(random.sample(overall2, num_to_select2))
    final_list.extend(random.sample(overall3, num_to_select3))
    final_list.extend(random.sample(overall4, num_to_select4))
    final_list.extend(random.sample(overall5, num_to_select5))
    return final_list

# Convert the lists in data frames, add the label for each category, split the train, test data and
# concatenate the results into a single data frame.

df = pd.DataFrame(calculate_percentage())
df_grocery['Label'] = "Grocery"

df.groupby('overall').count()

#Split randomly 'Beauty' into train and test sets
train_df_beauty = df.sample(frac = 0.7)
test_df_beauty = df.drop(train_df_beauty.index)

df_grocery = pd.DataFrame(calculate_percentage())
df_grocery['Label'] = 'Grocery'

#Split randomly 'Grocery' into train and test sets
train_df_grocery = df_grocery.sample(frac = 0.7)
test_df_grocery = df_grocery.drop(train_df_grocery.index)

df_movies_tv = pd.DataFrame(calculate_percentage())
df_movies_tv['Label'] = 'Movies & TV'

#Split randomly 'movies & tv' into train and test sets
train_df_movies_tv = df_movies_tv.sample(frac = 0.7)
test_df_movies_tv = df_movies_tv.drop(train_df_movies_tv.index)

df_toys_games = pd.DataFrame(calculate_percentage())
df_toys_games['Label'] = 'Toys & Games'
df_toys_games = df_toys_games.drop(df_toys_games.index[100000])

#Split randomly 'toys & games' into train and test sets
train_df_toys_games = df_toys_games.sample(frac = 0.7)
test_df_toys_games = df_toys_games.drop(train_df_toys_games.index)

df_train_total = pd.concat([train_df_beauty, train_df_grocery, train_df_movies_tv, train_df_toys_games])
df_test_total = pd.concat([test_df_beauty, test_df_grocery, test_df_movies_tv, test_df_toys_games])

# Create the pickle files (earese the spyder variable explorer and open the pickle file)

#TRAIN DATA
df_train_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/1 - df_train_total.pkl")
df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/1 - df_train_total.pkl")

#TEST DATA
df_test_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/1 - df_test_total.pkl")
df_test_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/1 - df_test_total.pkl")

#clean pontuation and signs from text
def standardize_text(df, text_field):
    df[text_field] = df[text_field].str.replace(r"http\S+", "")
    df[text_field] = df[text_field].str.replace(r"http", "")
    df[text_field] = df[text_field].str.replace(r"@\S+", "")
    df[text_field] = df[text_field].str.replace(r"[\,\$\&\*\%\(\)\~\-\"\;\^\+\#\/|0-9]", " ")
    df[text_field] = df[text_field].str.replace(r"@", "at")
    df[text_field] = df[text_field].str.lower()
    return df

df_train_total = standardize_text(df_train_total, 'reviewText')
df_train_total.get('reviewText').head()

# REMOVE SINGLE LETTERS
aux_single_letter =[[word for word in text.split() if len(word)>1] for text in df_train_total.get('reviewText')]
aux_single_letter = [ ' '.join(l) for l in aux_single_letter]
df_train_total['reviewText'] = aux_single_letter

# EXPAND CONTRACTIONS
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

clean_df_aux = []
for line in df_train_total.get('reviewText'):
    clean_df_aux.append(expand_contractions(line))

df_train_total['reviewText'] = clean_df_aux
df_train_total['reviewText'].head()

# REMOVING END CHARS
def standardize_text2(df, text_field):
    df[text_field] = df[text_field].str.replace(r"[\.\=\'\:\;\?\!\_\...]", " ") # add se quiser \'
    return df

standardize_text2(df_train_total, 'reviewText')
df_train_total['reviewText'].head()

# REMOVE MULTIPLE SPACES
df_aux = []
for text in df_train_total.get('reviewText'):
    df_aux.append(re.sub(' +',' ',text))

df_train_total['reviewText'] = df_aux
df_train_total['reviewText'].head()

# REMOVE REPEATED CHARACTERS
def remove_repeated_characters(tokens):
    repeat_pattern = re.compile(r'(\w*)(\w)\2(\w*)')
    match_substitution = r'\1\2\3'
    def replace(old_word):
        if wordnet.synsets(old_word):
            return old_word
        new_word = repeat_pattern.sub(match_substitution, old_word)
        if new_word != old_word:
            return replace(new_word)
        else:
            return new_word
    correct_tokens = [replace(word) for word in tokens]
    return correct_tokens

remove_repeated_chars = []
for text in df_train_total.get('reviewText'):
    txt = text.split()
    remove_repeated_chars.append(remove_repeated_characters(txt))

remove_repeated_chars = [ ' '.join(l) for l in remove_repeated_chars]

df_train_total['reviewText'] = remove_repeated_chars

# Create the pickle files (file with cleaned text)

#TRAIN DATA
df_train_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/2 - train total cleaned.pkl")
df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/2 - train total cleaned.pkl")

#TEST DATA
df_test_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/2 - test total cleaned.pkl")
df_test_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/2 - test total cleaned.pkl")

# Remove Stop Words
pathStopwords = "C:/Users/eduar/Documents/TXT Analytics/data/stop_words.txt"
# Read in and split the stopwords file.
with open(pathStopwords, 'r') as f:
    stop_words = f.read().split("\n")

dfList = df_train_total['reviewText'].tolist()
#LIST WITH NO STOPWORDS
my_new_list = [[word for word in text.split() if word not in stop_words] for text in dfList]
my_new_list = [ ' '.join(l) for l in my_new_list]
df_train_total['reviewText'] = my_new_list

# CORRECT SPELLING
# Count words to see the most common words in the dataset
def words(text): return re.findall(r'\w+', text.lower())

WORDS_train = []
for text in df_train_total_2.get('reviewText'):
    WORDS_train.extend(words(text))

from collections import Counter

count_words = Counter(WORDS_train)
#check the top 50 words used in all reviews
cenas = Counter(WORDS_train).most_common(20)
#number of unique words in all reviews
len(Counter(WORDS_train))



def P(word, N=sum(count_words.values())):
    "Probability of `word`."
    return count_words[word] / N

def correction(word):
    "Most probable spelling correction for word."
    return max(candidates(word), key=P)

def candidates(word):
    "Generate possible spelling corrections for word."
    return (known([word]) or known(edits1(word)) or known(edits2(word)) or [word])

def known(words):
    "The subset of `words` that appear in the dictionary of WORDS."
    return set(w for w in words if w in count_words)

def edits1(word):
    "All edits that are one edit away from `word`."
    letters    = 'abcdefghijklmnopqrstuvwxyz'
    splits     = [(word[:i], word[i:])    for i in range(len(word) + 1)]
    deletes    = [L + R[1:]               for L, R in splits if R]
    transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R)>1]
    replaces   = [L + c + R[1:]           for L, R in splits if R for c in letters]
    inserts    = [L + c + R               for L, R in splits for c in letters]
    return set(deletes + transposes + replaces + inserts)

def edits2(word):
    "All edits that are two edits away from `word`."
    return (e2 for e1 in edits1(word) for e2 in edits1(e1))

final_correction = [[correction(word) for word in text.split()]for text in df_train_before_stopwords['reviewText']]
final_correction = [ ' '.join(l) for l in final_correction]

df_train_total['reviewText'] = final_correction

#check correction function
correction('gamez')

#TRAIN DATA
df_train_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/3 - clean_spelling_pickle.pkl")
df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/3 - clean_spelling_pickle.pkl")

#TEST DATA
df_test_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/3 - test_clean_spelling_pickle.pkl")
df_test_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/3 - test_clean_spelling_pickle.pkl")

# LEMMATIZE
from nltk.stem import WordNetLemmatizer
wnl = WordNetLemmatizer()
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize

def pos_tag_text(text):
    # convert Penn treebank tag to wordnet tag

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    text = word_tokenize(text)
    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

# lemmatize text based on POS tags
def lemmatize_text(text):
    pos_tagged_text = pos_tag_text(text)
    lemmatized_tokens = [wnl.lemmatize(word, pos_tag) if pos_tag
                         else word
                         for word, pos_tag in pos_tagged_text]
    lemmatized_text = ' '.join(lemmatized_tokens)
    return lemmatized_text

clean_df_aux = []
for text in df_train_total.get('reviewText'):
    clean_df_aux.append(lemmatize_text(text))

# transform the auxiliar into the reviewtext lemmatized
df_train_total['reviewText'] = clean_df_aux

# Create the pickle file containing all the alteretains until this point
df_train_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")
df_test_total.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - test_clean_lemmatize.pkl")

## MODELS

#import for models
from collections import Counter
from sklearn.feature_extraction.text import CountVectorizer
import nltk
from nltk.collocations import BigramAssocMeasures, BigramCollocationFinder
from nltk.corpus import gutenberg
from operator import itemgetter
from copy import deepcopy
from nltk.util import ngrams
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.neighbors import KNeighborsClassifier
import matplotlib
import matplotlib.pyplot as plt
from __future__ import division
import math
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet as wn
from nltk import pos_tag, word_tokenize
import pandas as pd
import numpy as np
import re

df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")

# The code bellow was used to understand the Bag of Words, but during the report we decided to
# use another way to do this part, because it fits better with SVM

X_train = df_train_total["reviewText"]
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer = "word", max_features = 5000)
X_train_vectorized = vectorizer.fit_transform(X_train)
X_train_features = X_train_vectorized.todense()
X_train_vocab = vectorizer.vocabulary_

# Count words to see the most common words in the dataset
def words(text): return re.findall(r'\w+', text.lower())

WORDS_train = []
for text in df_train_total.get('reviewText'):
    WORDS_train.extend(words(text))

count_words = Counter(WORDS_train)
#check the top 50 words used in all reviews
cenas = Counter(WORDS_train).most_common(20)
#number of unique words in all reviews
len(Counter(WORDS_train))

count_vectorizer = CountVectorizer(stop_words='english')
#words, word_values = get_top_n_words(n_top_words=20, count_vectorizer=count_vectorizer, text_data=reindexed_data)
fig, ax = plt.subplots(figsize=(16,8))
ax.bar(range(len(words)), cenas)
ax.set_xticks(range(len(words)))
ax.set_xticklabels(words)
ax.set_title('Top Words')

#PLOT TOP 20 WORDS IN TRAINING SET
fig, ax = plt.subplots(figsize=(10,5))
ax.barh( range(len(cenas)), [t[1] for t in cenas] , height = 0.5 , align="center", color='#B2C5D8')
ax.set_yticks(range(len(cenas)))
ax.set_yticklabels(t[0] for t in cenas)
plt.title("Top 20 words in training set")
plt.tight_layout()
plt.show()

# BAG OF WORDS WITH NOUNS
wnl = WordNetLemmatizer()

def pos_tag_text(text):
    # convert Penn treebank tag to wordnet tag

    def penn_to_wn_tags(pos_tag):
        if pos_tag.startswith('J'):
            return wn.ADJ
        elif pos_tag.startswith('V'):
            return wn.VERB
        elif pos_tag.startswith('N'):
            return wn.NOUN
        elif pos_tag.startswith('R'):
            return wn.ADV
        else:
            return None
    text = word_tokenize(text)
    tagged_text = pos_tag(text)
    tagged_lower_text = [(word.lower(), penn_to_wn_tags(pos_tag))
                         for word, pos_tag in
                         tagged_text]
    return tagged_lower_text

#SIMPLE TEST
pos_tag_text('distribute')
pos_tag_text('give')


nouns = list()
nouns_all = list()
for text in df_train_total.get('reviewText'):
    pos_tagged_text = pos_tag_text(text)
    nouns.clear()
    for ( word, pos) in pos_tagged_text:
        if pos == 'N' or pos == 'n':
            nouns.append(word)
    nouns_all.append(' '.join(nouns))


df_nouns = df_train_total
df_nouns['reviewText'] = nouns_all

# SAVE THE BAG OF WORDS WITH NOUNS
df_nouns.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df nouns.pkl")
df_nouns = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df nouns.pkl")

# VERIFYING THE BIGRAMS
df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")

# FIRST ALTERNATIVE FOR BIGRAMS
df_aux = []
def words(text): return re.findall(r'\w+', text.lower())

for text in df_train_total.get('reviewText'):
    df_aux.extend(words(text))

# A SECOND ALTERNATIVE THAT PUT THE BIGRAMS INTO A DATA FRAME
bigrams= list()
bigrams_all = list()
for text in df_train_total.get('reviewText'):
    bigrams.clear()
    for word in text.split():
        bigrams.append(word)

    bigrams = list(ngrams(bigrams, 2))
    bigrams_all.append(tuple(bigrams))

#antes
bigrams = ngrams(df_aux, 2)
BigramFreq = Counter(bigrams_all)
get_bigrams_to_list = list(BigramFreq)

#tentative altrnativa de lista
get_bigrams_to_list = list(bigrams_all)
df_bigrams = df_train_total
df_bigrams['reviewText'] = get_bigrams_to_list

# MOST FREQUENT BIGRAMS
get_bigrams = BigramFreq.most_common(10)

# CREATE A PICKLE FILE TO STORE THE BIGRAMS ON THE DATASET
df_bigrams.to_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df bigrams.pkl")
df_bigrams = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df bigrams.pkl")

for text in df_bigrams.get('reviewText'):
    #for l in text:
    tuples_list = list(text)

# PLOT THE MOST FREQUENT BIGRAMS
fig, ax = plt.subplots(figsize=(10,5))
ax.barh( range(len(get_bigrams)), [t[1] for t in get_bigrams] , height = 0.5 , align="center", color='#B2C5D8')
ax.set_yticks(range(len(get_bigrams)))
ax.set_yticklabels([t[0] for t in get_bigrams])
plt.tight_layout()
plt.show()

# TF-IDF
tokenize = lambda doc: doc.lower().split(" ")

def jaccard_similarity(query, document):
    intersection = set(query).intersection(set(document))
    union = set(query).union(set(document))
    return len(intersection)/len(union)

def term_frequency(term, tokenized_document):
    return tokenized_document.count(term)

def sublinear_term_frequency(term, tokenized_document):
    count = tokenized_document.count(term)
    if count == 0:
        return 0
    return 1 + math.log(count)

def augmented_term_frequency(term, tokenized_document):
    max_count = max([term_frequency(t, tokenized_document) for t in tokenized_document])
    return (0.5 + ((0.5 * term_frequency(term, tokenized_document))/max_count))

def inverse_document_frequencies(tokenized_documents):
    idf_values = {}
    all_tokens_set = set([item for sublist in tokenized_documents for item in sublist])
    for tkn in all_tokens_set:
        contains_token = map(lambda doc: tkn in doc, tokenized_documents)
        idf_values[tkn] = 1 + math.log(len(tokenized_documents)/(sum(contains_token)))
    return idf_values

def tfidf(documents):
    tokenized_documents = [tokenize(d) for d in documents]
    idf = inverse_document_frequencies(tokenized_documents)
    tfidf_documents = []
    for document in tokenized_documents:
        doc_tfidf = []
        for term in idf.keys():
            tf = sublinear_term_frequency(term, document)
            doc_tfidf.append(tf * idf[term])
        tfidf_documents.append(doc_tfidf)
    return tfidf_documents

#in Scikit-Learn
sklearn_tfidf = TfidfVectorizer(norm='l2',min_df=0, use_idf=True, smooth_idf=False, sublinear_tf=True, tokenizer=tokenize)
sklearn_representation = sklearn_tfidf.fit_transform(all_documents)
print(sklearn_representation) # RESULT (document, word) tfidf value

## SVD

lsa =TruncatedSVD(n_components=100)
#bow normal and nouns
lsa.fit(X)
reviews_concepts_matrix =lsa.fit_transform(X)

#bigrams
lsa.fit(bow_train_features)
reviews_concepts_matrix =lsa.fit_transform(bow_train_features)

comps = lsa.components_

#Gives each concept with 10 words associated
terms = vectorizer.get_feature_names()
for i, comp in enumerate(lsa.components_):
    termsInComp = zip(terms,comp)
    sortedTerms = sorted(termsInComp, key=lambda x: x[1], reverse = True) [:10]
    print("Concept %d: " % i)
    for term in sortedTerms:
        print(term[0])
    print(" ")

# PLOT ELBOW
explained_variance = lsa.explained_variance_ratio_
print(lsa.explained_variance_ratio_)

singular_values = lsa.singular_values_

explained_variance_ratio_plot = np.cumsum(explained_variance)
plt.plot(explained_variance_ratio_plot )

plt.plot(singular_values)
plt.ylabel("Singular Values")
plt.xlabel("Concepts")

comps_reduzida = comps[:10][:]

# CATEGORIES LIST
categories_target = ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]
categories_target_nr = [0,1,2,3]
# FOR BOW - concepts list
concepts_list = ["Concept 1", "Concept 2", "Concept 3", "Concept 4", "Concept 5", "Concept 6", "Concept 7", "Concept 8"]
# FOR BOW NOUNS - concepts list
concepts_list = ["Concept 1", "Concept 2", "Concept 3", "Concept 4", "Concept 5", "Concept 6", "Concept 7", "Concept 8", "Concept 9", "Concept 10"]

df_reviews_concepts=pd.DataFrame(reviews_concepts_matrix)

# FOR BOW
new_list_labels = [word for word in df_train_total.get('Label')]
# FOR BOW NOUNS
new_list_labels = [word for word in df_nouns.get('Label')]

df_reviews_concepts['Label'] =  new_list_labels

df_teste =(df_reviews_concepts.groupby(['Label']).mean())

df_categories_concepts = df_teste.reset_index()
# for BOW normal
new_df =df_teste.drop(df_teste.columns[8:100], axis=1)
# for NOUNS
new_df =df_teste.drop(df_teste.columns[10:100], axis=1)

# PLOT THE GRID OF WEIGHTS OF EACH CONCEPT TO EACH CATEGORY
fig, ax = plt.subplots()
im = ax.imshow(new_df)

# We want to show all ticks...
ax.set_xticks(np.arange(len(concepts_list)))
ax.set_yticks(np.arange(len(categories_target)))
# ... and label them with the respective list entries
ax.set_xticklabels(concepts_list)
ax.set_yticklabels(categories_target)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

new_df_aux = new_df.reset_index()
new_df_aux =new_df_aux.drop(columns = ['Label'])


for i in range(len(categories_target)):
    for j in range(len(concepts_list)):
        new_df_aux.iat[i,j] ="%.3f" % new_df_aux.iat[i,j]
        text = ax.text(j, i, str(new_df_aux.iat[i,j]), ha="center", va="center", color="w")

ax.set_title("Catgories vs Concepts - Bag of words of nouns")
plt.figure(figsize=(10,5))
fig.tight_layout()
plt.show()

## Bag Of Words treated to Support Vector Machines

# IMPORTS
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
import itertools

df_train_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - clean_lemmatize.pkl")
df_test_total = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/4 - test_clean_lemmatize.pkl")
#TRAIN CORPUS - for bigrams and bow normal
train_corpus = df_train_total['reviewText'].tolist()
train_label = df_train_total['Label'].tolist()
#TEST CORPUS - for the three models
test_corpus = df_test_total['reviewText'].tolist()
test_label = df_test_total['Label'].tolist()

#nouns corpus train
df_nouns = pd.read_pickle("C:/Users/eduar/Documents/TXT Analytics/data/pickle/5 - df nouns.pkl")

train_corpus_nouns = df_nouns['reviewText'].tolist()
train_label_nouns = df_nouns['Label'].tolist()

#STEP 1 - CHOOSE ONE OF THE MODELS TO RUN
#-----------------------------------------------BOW normal
def bow_extractor(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()

vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW bigrams
def bow_extractor_bigrams(corpus, ngram_range=(2,2)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor_bigrams(train_corpus)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
vocab = bow_vectorizer.get_feature_names()

#-----------------------------------------------BOW Nouns
def bow_extractor_nouns(corpus, ngram_range=(1,1)):

    vectorizer = CountVectorizer(analyzer = "word",
                                 max_features = 5000,
                                 ngram_range=ngram_range)
    features = vectorizer.fit_transform(corpus)
    return vectorizer, features

# bag of words features
bow_vectorizer, bow_train_features = bow_extractor_nouns(train_corpus_nouns)
bow_test_features = bow_vectorizer.transform(test_corpus)
bow_matrix = bow_test_features.todense()
vocab = bow_vectorizer.get_feature_names()

#--------------------------------------------------------------
#STEP 2 - EQUAL FOR ALL THREE MODELS

def get_metrics(true_labels, predicted_labels):

    print ('Accuracy:', np.round(
                        metrics.accuracy_score(true_labels,
                                               predicted_labels),
                        2))
    print ('Precision:', np.round(
                        metrics.precision_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('Recall:', np.round(
                        metrics.recall_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))
    print ('F1 Score:', np.round(
                        metrics.f1_score(true_labels,
                                               predicted_labels,
                                               average='weighted'),
                        2))


def train_predict_evaluate_model(classifier,
                                 train_features, train_labels,
                                 test_features, test_labels):
    # build model
    classifier.fit(train_features, train_labels)
    # predict using model
    predictions = classifier.predict(test_features)
    # evaluate model prediction performance
    get_metrics(true_labels=test_labels,
                predicted_labels=predictions)
    return predictions

mnb = MultinomialNB()
svm = LinearSVC(penalty='l2', C=1.0) # THE PARAMETERS FROM HANDOUT

# Support Vector Machine with bag of words features
svm_bow_predictions = train_predict_evaluate_model(classifier=svm,
                                           train_features=bow_train_features,
                                           train_labels=train_label,
                                           test_features=bow_test_features,
                                           test_labels=test_label)

cm = metrics.confusion_matrix(test_label, svm_bow_predictions)
pd.DataFrame(cm, index=range(0,4), columns=range(0,4))

#----------------------------------------------- PLOT
X_scaled = preprocessing.scale(cm)

df_cm = pd.DataFrame(cm, index = [i for i in ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]],
                  columns = [i for i in ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]])
plt.figure(figsize = (10,7))
sns.heatmap(df_cm, annot=True, cmap ="Blues")
sns.palplot(sns.color_palette("GnBu_d"))

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        cm[i,j] ="%.3f" % cm.iat[i,j]
        plt.text(j, i, cm[i, j], horizontalalignment="center",color="white" if cm[i, j] > thresh else "black")
#--------------END OF PLOT FUNCTION

categories_target = ["Beauty", "Grocery", "Movies & TV", "Toys & Games"]

class_names = categories_target

plt.figure(figsize = (10,7))
plot_confusion_matrix(cm, classes=class_names, normalize=True, title='Normalized confusion matrix')
plt.show()


# Import necessary File
import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import json
import pickle
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.optimizers import SGD
import random
# Load necessary file
words=[]
classes = []
documents = []
ignore_words = ['?', '!']
# Intents.json – The data file which has predefined patterns and responses.
data_file = open('intents.json').read()
intents = json.loads(data_file)
# Preprocessing of data
for intent in intents['intents']:
    for pattern in intent['patterns']:
        #tokenize each word
        w = nltk.word_tokenize(pattern)
        words.extend(w)
        #add documents in the corpus
        documents.append((w, intent['tag']))
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])
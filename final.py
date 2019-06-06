#!/usr/bin/env python
from get_api import getres
import numpy as np
import sys
import os
import nltk
from nltk.corpus import stopwords
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# import keras
import json
# BASE_URL_1 = "./training/celebrityDataset/"
# BASE_URL_2 = "./training/fakeNewsDataset/"
# PATH1 = "./ibm-api-connect/natlang_responses/celebrityDataset/"
# PATH2 = "./ibm-api-connect/tone_responses/celebrityDataset/"
# PATH21 = "./ibm-api-connect/natlang_responses/fakeNewsDataset/"
# PATH22 = "./ibm-api-connect/tone_responses/fakeNewsDataset/"

print("loading data...")

print("loading file embeddings...")
f = open('glove.6B.100d.txt')
embeddings_index = dict()
for line in f:
   values = line.split()
   word = values[0]
   coefs = np.asarray(values[1:], dtype='float32')
   embeddings_index[word] = coefs
f.close()

print("loaded {} file embeddings".format(len(embeddings_index.keys())))
stop_words = {}
def boolean_indexing(v, fillval=0.0):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out

def predict(res):
   sentiment, concepts, keywords, tones, text = res
   my_embeddings = np.zeros((100))
   total_relevance = np.array([([keyword[1] for keyword in keywords])])
   print(total_relevance)
   total_relevance.resize(5)
   print(total_relevance)
   my_concepts = np.zeros((100))
   for concept_list in concepts:
      temp_concept = np.zeros((100))
      for concept in nltk.word_tokenize(concept_list[0])[-1:]:
         temp_concept +=embeddings_index.get(concept.lower(), np.zeros((100)))
      my_concepts+=temp_concept*concept_list[1]
   my_tones = np.zeros((100))
   for tone in tones:
      my_tones+=embeddings_index.get(tone[0],np.zeros((100)))*tone[1]
   tokenized = nltk.word_tokenize(text)
   for j in tokenized:
      if j.lower() in stop_words:
         continue
      my_embeddings+=embeddings_index.get(j.lower(), np.zeros((100)))
   total_as_vector = np.array((np.concatenate((my_embeddings,my_tones,np.array(sentiment),my_concepts,total_relevance), axis=None)))
   return total_as_vector

#logreg = LogisticRegression(solver = 'lbfgs',C=10**(-2))
import pickle
logreg = None
with open("./weights","rb") as file_:
   logreg = pickle.load(file_)
res = getres(sys.argv[1])
input_ = predict(res)
input_ = np.reshape(input_,(306,1))
ans = round(logreg.predict(input_.T)[0])
print(ans)
sys.exit(1-int(ans)) 

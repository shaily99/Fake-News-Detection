#!/usr/bin/env python


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
# from keras.models import Sequential
# from keras.layers import Dense, GRU, LSTM, Dropout, Bidirectional, Embedding
# from keras.callbacks import LambdaCallback, ModelCheckpoint, EarlyStopping
# from keras.preprocessing.text import Tokenizer
# from keras.preprocessing.sequence import pad_sequences

BASE_URL_1 = "./training/celebrityDataset/"
BASE_URL_2 = "./training/fakeNewsDataset/"
PATH1 = "./ibm-api-connect/natlang_responses/celebrityDataset/"
PATH2 = "./ibm-api-connect/tone_responses/celebrityDataset/"
PATH21 = "./ibm-api-connect/natlang_responses/fakeNewsDataset/"
PATH22 = "./ibm-api-connect/tone_responses/fakeNewsDataset/"

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


# t = Tokenizer()
# """
# for fileid in nltk.corpus.reuters.fileids():
#   t.fit_on_texts(nltk.corpus.reuters.words(fileid))
#   print("{} Done".format(fileid))
# """
# t.fit_on_texts(embeddings_index.keys())
# vocab_size=len(t.word_index) + 1
# embedding_matrix = np.zeros((vocab_size, 100))
# for word, i in t.word_index.items():
#   embedding_vector = embeddings_index.get(word)
#   if embedding_vector is not None:
#     embedding_matrix[i] = embedding_vector


print("loaded {} file embeddings".format(len(embeddings_index.keys())))
def load_fromdir(BASE_URL, PATH_TO_API, PATH2):
    file_list = os.listdir(BASE_URL)
    res_list = []
    sentiment_list = []
    concept_list = []
    keywords_list = []
    tone_list = []
    for fileid in file_list:
        file_name = "".join(fileid.split(".")[:-1])+".json"
        api_file = open(PATH_TO_API+file_name)
        api_json = json.load(api_file)
        sentiment = api_json["sentiment"]["document"]["score"]
        concepts = [(concepts["text"], concepts["relevance"]) for concepts in api_json["concepts"]]
        categories  = (api_json["categories"][0]["label"].split("/"),api_json["categories"][0]["score"])
        keywords = [(keywords["text"],keywords["relevance"]) for keywords in api_json["keywords"]]
        api_file.close()
        api_file = open(PATH2+file_name)
        api_json = json.load(api_file)
        tones = [(tone["tone_id"],tone["score"]) for tone in api_json["document_tone"]["tones"]]
        api_file.close
        opened = open(BASE_URL+fileid)
        text = opened.read()
        res_list.append(text)
        sentiment_list.append(sentiment)
        concept_list.append(concepts)
        tone_list.append(tones)
        keywords_list.append(keywords)
        opened.close()
    print(len(res_list))
    return res_list, concept_list, sentiment_list, tone_list, keywords_list

fake_list,fake_concept_list, fake_sentiment_list, fake_tone_list, fake_keywords_list = load_fromdir(BASE_URL_1+"fake/",PATH1+"fake/", PATH2+"fake/")
real_list,real_concept_list, real_sentiment_list, real_tone_list, real_keywords_list = load_fromdir(BASE_URL_1+"legit/",PATH1+"legit/", PATH2+"legit/")
fake_list2,fake_concept_list2, fake_sentiment_list2, fake_tone_list2, fake_keywords_list2 = load_fromdir(BASE_URL_2+"fake/",PATH21+"fake/", PATH22+"fake/")
real_list2,real_concept_list2, real_sentiment_list2, real_tone_list2, real_keywords_list2 = load_fromdir(BASE_URL_2+"legit/",PATH21+"legit/", PATH22+"legit/")

fake_list.extend(fake_list2)
fake_concept_list.extend(fake_concept_list2)
fake_sentiment_list.extend(fake_sentiment_list2)
fake_tone_list.extend(fake_tone_list2)
fake_keywords_list.extend(fake_keywords_list2)

real_list.extend(real_list2)
real_concept_list.extend(real_concept_list2)
real_sentiment_list.extend(real_sentiment_list2)
real_tone_list.extend(real_tone_list2)
real_keywords_list.extend(real_keywords_list2)

print("loaded!")
total = real_list+fake_list
total_concept = real_concept_list+fake_concept_list
total_tone = real_tone_list+fake_tone_list
total_sentiment = real_sentiment_list+fake_sentiment_list
total_keywords = real_keywords_list+fake_keywords_list

targets = [1 for i in real_list]+[0 for i in fake_list]
print("preprocessing")
def boolean_indexing(v, fillval=0.0):
    lens = np.array([len(item) for item in v])
    mask = lens[:,None] > np.arange(lens.max())
    out = np.full(mask.shape,fillval)
    out[mask] = np.concatenate(v)
    return out


stop_words = set() # set(stopwords.words('english'))
total_as_vectors = []
total_relevance = []
for i in range(len(total)):
    my_embeddings = np.zeros((100))
    total_relevance.append([keyword[1] for keyword in total_keywords[i]])
    my_concepts = np.zeros((100))
    for concept_list in total_concept[i]:
        temp_concept = np.zeros((100))
        for concept in nltk.word_tokenize(concept_list[0])[-1:]:
            temp_concept +=embeddings_index.get(concept.lower(), np.zeros((100)))
        my_concepts+=temp_concept*concept_list[1]
    my_tones = np.zeros((100))
    for tone in total_tone[i]:
        my_tones+=embeddings_index.get(tone[0],np.zeros((100)))*tone[1]
    tokenized = nltk.word_tokenize(total[i])
    pos = nltk.pos_tag(tokenized)
    adjectives = 0
    for k,j in enumerate(tokenized):
        if j.lower() in stop_words:
            continue
        if "JJ" in pos[k]:
            adjectives +=1
        my_embeddings+=embeddings_index.get(j.lower(), np.zeros((100)))
    
total_as_vectors.append(np.concatenate((my_embeddings,my_tones,np.array(total_sentiment[i]),my_concepts,np.array([adjectives])), 
axis=None))
    # if np.isnan(total_as_vectors[-1]).any():
    #     print("Caught a NaN at {}".format(i))
    #     sys.exit(1)

total_relevance = boolean_indexing(total_relevance)
total_as_vectors = np.concatenate((total_as_vectors,total_relevance), axis=1)
print("loading model")


X_train, X_test, y_train, y_test = train_test_split(total_as_vectors, targets, random_state=7)
logreg = LogisticRegression(solver = 'lbfgs',C=10**(-2))
logreg.fit(X_train, y_train)

print("Train accuracy {}".format(logreg.score(X_train,y_train)))
print("Test accuracy {}".format(logreg.score(X_test,y_test)))


# model = Sequential()
# model.add(Embedding(vocab_size, 100,weights=[embedding_matrix], trainable=False))
# model.add(Bidirectional(LSTM(64,stateful=False)))
# model.add(Dropout(0.4))
# model.add(Dense(1,activation="sigmoid"))
# model.compile(loss='binary_crossentropy', optimizer="adam", metrics=['accuracy'])
# model.load_weights("./rnn-64-1-20-0.68.hdf5")
# rnn_test = np.array(pad_sequences(total))
# X_train, X_test, y_train, y_test = train_test_split(rnn_test, targets, random_state=7)
# model.evaluate(X_train, y_train)

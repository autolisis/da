#! /usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2018 g <g@ABCL>
#
# Distributed under terms of the MIT license.
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import re
import sys
import pickle

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk as nlp


df = pd.read_csv("brand-classifier.csv", encoding="latin1")

work_data = pd.DataFrame()
work_data["tweet"] = df.description
work_data["gender"] = df.gender
work_data.dropna(inplace=True)

brandMask = work_data['gender'] == 'brand'
notbrandMask = work_data['gender'] != 'brand'
brandRows = work_data[brandMask]
notbrandRows = work_data[notbrandMask]

brandRows.gender = 1
notbrandRows.gender = 0


frames = [brandRows, notbrandRows]
data = pd.concat(frames, ignore_index=True)

# Clean data Drop nulls
data.dropna(inplace=True)

# lemma = nlp.WordNetLemmatizer()

# # Clean up unnecessary characters and remove stop words
# tweets_list = []            # empty list
# for each in data.tweet:
#     # regex to clean unnecesarry chars
#     each = re.sub("[^a-zA-Z]", " ", str(each))
#     # lowercase all
#     each = each.lower()
#     # split all by tokenizing
#     each = nlp.word_tokenize(each)
#     # delete stop words from your array
#     each = [word for word in each if not word in set(
#         stopwords.words("english"))]
#     # lemmatize "memories" -> "memory"
#     each = [lemma.lemmatize(word) for word in each]
#     # make them one string again
#     each = " ".join(each)
#     # put them into big array
#     tweets_list.append(each)

# with open('temp', 'wb') as f:
#     pickle.dump(tweets_list, f)
with open('temp', 'rb') as f:
    tweets_list = pickle.load(f)

count_vectorizer = CountVectorizer(stop_words="english", max_features=500)

matrix = count_vectorizer.fit_transform(tweets_list).toarray()
words = count_vectorizer.get_feature_names()

y = data.gender.values
x = matrix


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
rfc = RandomForestClassifier(n_estimators=100)

rfc.fit(train_x, train_y)

rfc.score(test_x, test_y)
y_head_ml = rfc.predict(test_x)

cm = confusion_matrix(test_y, y_head_ml)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, cbar=False, annot=True, cmap="Blues", fmt=".2f")
plt.show()

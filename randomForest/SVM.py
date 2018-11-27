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


#data preprocessing 
df = pd.read_csv("brand-classifier.csv", encoding="latin1")
work_data= df[['description','gender']]
work_data.rename(columns={'description':'tweet'},inplace=True)
work_data['filtered'] = work_data.gender.apply(lambda x: 1 if x=='brand' else 0)
work_data.dropna(inplace=True)

#generation of features  
count_vectorizer = CountVectorizer(stop_words="english", max_features=500)
with open('temp', 'rb') as f:
    tweets_list = pickle.load(f)
matrix = count_vectorizer.fit_transform(tweets_list).toarray()
words = count_vectorizer.get_feature_names()

y = data.gender.values
x = matrix


train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)

from sklearn.svm import SVC
clf = SVC(gamma='auto')
clf.fit(train_x,train_y)

clf.score(test_x, test_y)

y_pred = clf.predict(test_x)

cm = confusion_matrix(test_y, y_pred)
cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
sns.heatmap(cm, cbar=False, annot=True, cmap="Blues", fmt=".2f")
# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
%matplotlib inline


# importing data
df = pd.read_csv("scripts.csv")
del df["Unnamed: 0"]

dial_df = df.drop(["EpisodeNo","SEID","Season"],axis=1)

dial_df = dial_df[(dial_df["Character"]=="JERRY") | (dial_df["Character"]=="ELAINE") | (dial_df["Character"]=="GEORGE") | (dial_df["Character"]=="KRAMER")]

dial_df["Dialogue"][0]

from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)

def text_process(dialogue):
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split()]

X = dial_df["Dialogue"]
y = dial_df["Character"]

from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(analyzer=text_process).fit(X)

print(len(vectorizer.vocabulary_))

X = vectorizer.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.ensemble import RandomForestClassifier as RFC
rfc = RFC(n_estimators=50,criterion='entropy',random_state=0, n_jobs=-1)
rfc.fit(X_train, y_train)

predict = rfc.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))

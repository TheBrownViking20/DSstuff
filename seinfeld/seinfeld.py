# Importing libraries
import numpy as np
import pandas as pd

# importing data
df = pd.read_csv("scripts.csv")
del df["Unnamed: 0"]

dial_df = df.drop(["EpisodeNo","SEID","Season"],axis=1)

dial_df = dial_df[(dial_df["Character"]=="ELAINE") | (dial_df["Character"]=="GEORGE") | (dial_df["Character"]=="KRAMER")]

dial_df["Dialogue"][4]

from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)

def text_process(dialogue):
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=''.join(nopunc)
    return [word for word in nopunc.split()]

X = dial_df["Dialogue"]
y = dial_df["Character"]

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(analyzer=text_process).fit(X)

print(len(vectorizer.vocabulary_))
X = vectorizer.transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=101)

from sklearn.naive_bayes import MultinomialNB as MNB
from sklearn.linear_model import LogisticRegression as LR
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import VotingClassifier as VC
mnb = MNB(alpha=10)
lr = LR(random_state=101)
rfc = RFC(n_estimators=80, criterion="entropy", random_state=42, n_jobs=-1)
clf = VC(estimators=[('mnb', mnb), ('lr', lr), ('rfc', rfc)], voting='hard')

clf.fit(X_train,y_train)

predict = clf.predict(X_test)

from sklearn.metrics import confusion_matrix, classification_report
print(confusion_matrix(y_test, predict))
print('\n')
print(classification_report(y_test, predict))


def predictor(s):
    s = vectorizer.transform(s)
    pre = clf.predict(s)
    print(pre)

predictor(['I\'m on the Mexican, whoa oh oh, radio.'])
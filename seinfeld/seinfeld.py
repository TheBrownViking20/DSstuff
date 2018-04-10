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


dial_df["Character"].value_counts().head(12).plot(kind="bar")

def corpus_creator(name):
    st = "" 
    for i in dial_df["Dialogue"][dial_df["Character"]==name]:
        st = st + i
    return st

corpus_df = pd.DataFrame()
corpus_df["Character"] = list(dial_df["Character"].value_counts().head(12).index)

li = []
for i in corpus_df["Character"]:
    li.append(corpus_creator(i))

corpus_df["Dialogues"] = li

from sklearn.feature_extraction import text
punc = ['.', ',', '"', "'", '?', '!', ':', ';', '(', ')', '[', ']', '{', '}',"%"]
stop_words = text.ENGLISH_STOP_WORDS.union(punc)

from nltk.tokenize import word_tokenize
def text_process(dialogue):
    dialogue = word_tokenize(dialogue)
    nopunc=[word.lower() for word in dialogue if word not in stop_words]
    nopunc=' '.join(nopunc)
    return [word for word in nopunc.split()]

corpus_df["Dialogues"] = corpus_df["Dialogues"].apply(lambda x: text_process(x))

corpus_df["Length"] = corpus_df["Dialogues"].apply(lambda x: len(x))

fig, ax = plt.subplots(figsize=(10,10))
sns.barplot(ax=ax,y="Length",x="Character",data=corpus_df)

import gensim

dictionary = gensim.corpora.Dictionary(corpus_df["Dialogues"])
print(dictionary[567])
print(dictionary.token2id['cereal'])
print("Number of words in dictionary: ",len(dictionary))

corpus = [dictionary.doc2bow(bw) for bw in corpus_df["Dialogues"]]
print(corpus)

tf_idf = gensim.models.TfidfModel(corpus)

sims = gensim.similarities.Similarity('/output',tf_idf[corpus],num_features=len(dictionary))

sim_list = []
for i in range(12):
    query = dictionary.doc2bow(corpus_df["Dialogues"][i])
    query_tf_idf = tf_idf[query]
    sim_list.append(sims[query_tf_idf])
    
corr_df = pd.DataFrame()
j=0
for i in corpus_df["Character"]:
    corr_df[i] = sim_list[j]
    j = j + 1    
    
fig, ax = plt.subplots(figsize=(12,12))
sns.heatmap(corr_df,ax=ax,annot=True)
ax.set_yticklabels(corpus_df.Character)
plt.show()
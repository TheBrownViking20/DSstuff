import nltk
from nltk.tokenize import word_tokenize
import random
all_words = []

short_pos = open("short_reviews/positive.txt","r").read()
short_neg = open("short_reviews/negative.txt","r").read()

documents = []

for r in short_pos.split('\n'):
    documents.append((r,"pos"))
    
for r in short_pos.split('\n'):
    documents.append((r,"neg"))
    
short_pos_words = word_tokenize(short_pos)
short_neg_words = word_tokenize(short_neg)

for w in short_pos_words:
    all_words.append(w.lower())

for w in short_neg_words:
    all_words.append(w.lower())

# Words as features

all_words = nltk.FreqDist(all_words)
word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

featuresets = [(find_features(rev),category) for (rev,category) in documents]

random.shuffle(featuresets)

# Naive bayes
training_set = featuresets[:6000]
testing_set = featuresets[6000:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy  percent:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

# scikit learn incorporation
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

MNB_Classifier = SklearnClassifier(MultinomialNB())
MNB_Classifier.train(training_set)
print("MNB_Classifier accuracy percent: ",(nltk.classify.accuracy(MNB_Classifier,testing_set))*100)

#GNB_Classifier = SklearnClassifier(GaussianNB())
#GNB_Classifier.train(training_set)
#print("GNB_Classifier accuracy percent: ",(nltk.classify.accuracy(GNB_Classifier,testing_set))*100)

BNB_Classifier = SklearnClassifier(BernoulliNB())
BNB_Classifier.train(training_set)
print("BNB_Classifier accuracy percent: ",(nltk.classify.accuracy(BNB_Classifier,testing_set))*100)
    
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

LogisticRegression_Classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_Classifier.train(training_set)
print("LogisticRegression accuracy percent: ",(nltk.classify.accuracy(LogisticRegression_Classifier,testing_set))*100)
    
SVC_Classifier = SklearnClassifier(SVC())
SVC_Classifier.train(training_set)
print("SVC accuracy percent: ",(nltk.classify.accuracy(SVC_Classifier,testing_set))*100)

SGDClassifier_Classifier = SklearnClassifier(SGDClassifier())
SGDClassifier_Classifier.train(training_set)
print("SGDClassifier accuracy percent: ",(nltk.classify.accuracy(SGDClassifier_Classifier,testing_set))*100)
    
LinearSVC_Classifier = SklearnClassifier(LinearSVC())
LinearSVC_Classifier.train(training_set)
print("LinearSVC accuracy percent: ",(nltk.classify.accuracy(LinearSVC_Classifier,testing_set))*100)

NuSVC_Classifier = SklearnClassifier(NuSVC())
NuSVC_Classifier.train(training_set)
print("NuSVC accuracy percent: ",(nltk.classify.accuracy(NuSVC_Classifier,testing_set))*100)

# Combining algos with a vote
from nltk.classify import ClassifierI
from statistics import mode

class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
        
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
    
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        
        choice_votes = votes.count(mode(votes))
        conf = choice_votes / len(votes)
        return conf
    
voted_classifier = VoteClassifier(classifier,
                                  NuSVC_Classifier,
                                  LinearSVC_Classifier,
                                  SGDClassifier_Classifier,
                                  MNB_Classifier,
                                  BNB_Classifier,
                                  LogisticRegression_Classifier)
print("voted_classifier accuracy percent: ",(nltk.classify.accuracy(voted_classifier,testing_set))*100)
# Official library for NLP in python
import nltk

nltk.download()


#Tokenizing - separating by words and sentences
from nltk.tokenize import sent_tokenize, word_tokenize
example_text = "Trust me Mr. Smith, I don't give a damn. About waht you ask? Well, about the quick brown fox jumping over the lazy dog."
print(sent_tokenize(example_text))
print(word_tokenize(example_text))

for i in word_tokenize(example_text):
    print(i)
    
# Stop words - words not required oor words that hamper the dataset
from nltk.corpus import stopwords
example_sentence = "This is an example showing off stop word filtration."
stop_words = set(stopwords.words("english"))
print(stop_words)

filtered_sentence = [w for w in words if not w in stop_words]
print(filtered_sentence)

# Stemming - taking stem of the words
from nltk.stem import PorterStemmer
ps = PorterStemmer()
eg = ["python","pythoner","pythoning","pythoned","pythonly"]
for w in eg:
    print(ps.stem(w))
new = "It is very import to be pythonly while you are pythoning with python. All pythoners have pythoned."
words = word_tokenize(new)
for w in words:
    print(ps.stem(w))
    
# Speech tagging
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer

train_text = state_union.raw("2005-GWBush.txt")
sample_text = state_union.raw("2006-GWBush.txt")

custom_sent_tokenizer = PunktSentenceTokenizer(train_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            tagged = nltk.pos_tag(words)
            print(tagged)
    except Exception as e:
        print(str(e))
 
process_content()   

# Chunking

def process_content_two():
    for i in tokenized:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        chunkGram = r"""Chunk: {<RB.?>*<VB.?><NNP>+<NN>?} """
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        chunked.draw()            
process_content_two()

# Chinking

def process_content_three():
    for i in tokenized[5:]:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        chunkGram = r"""Chunk: {<.*>+}
                                }<VB.?|IN|DT|TO>+{ """
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        chunked.draw()            
process_content_three()

# Named entity recognition

def process_content_four():
    for i in tokenized[5:]:
        words = nltk.word_tokenize(i)
        tagged = nltk.pos_tag(words)
        namedEnt = nltk.ne_chunk(tagged,binary=True)
        namedEnt.draw()
process_content_four()

# Lemmatizing
import nltk
from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()
print(lemmatizer.lemmatize("cats"))
print(lemmatizer.lemmatize("cacti"))
print(lemmatizer.lemmatize("geese"))
print(lemmatizer.lemmatize("rocks",pos="v"))
print(lemmatizer.lemmatize("python"))
print(lemmatizer.lemmatize("better",pos="a"))

# Corpora
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

sample = gutenberg.raw("bible-kjv.txt")
tok = sent_tokenize(sample)
print(tok[5:15])

# Wordnet
from nltk.corpus import wordnet

syns = wordnet.synsets("program")

print(syns[0].lemmas())

print(syns[0].definition())

print(syns[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
print(set(synonyms))
print(set(antonyms))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("boat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("car.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("ship.n.01")
w2 = wordnet.synset("cat.n.01")
print(w1.wup_similarity(w2))

w1 = wordnet.synset("iron.n.01")
w2 = wordnet.synset("copper.n.01")
print(w1.wup_similarity(w2))

# Text Classification
import nltk
import random
from nltk.corpus import movie_reviews

documents = [(list(movie_reviews.words(fileid)),category) for category in movie_reviews.categories() for fileid in movie_reviews.fileids(category)]
random.shuffle(documents)
print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())

all_words = nltk.FreqDist(all_words)
print(all_words.most_common(15))

# Words as features

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
    return features

print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev),category) for (rev,category) in documents]

# Naive bayes

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

classifier = nltk.NaiveBayesClassifier.train(training_set)
print("Naive Bayes Algo accuracy  percent:",(nltk.classify.accuracy(classifier,testing_set))*100)
classifier.show_most_informative_features(15)

# Save classifier with pickle
import pickle
s = open("naivebayes.pickle","wb")
pickle.dump(classifier,s)
s.close()

d = open("naivebayes.pickle","rb")
classifier = pickle.load(d)
d.close()

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

print("Classification:",voted_classifier.classify(testing_set[0][0]),"Confidence %:",voted_classifier.confidence(testing_set[0][0])*100)
print("Classification:",voted_classifier.classify(testing_set[1][0]),"Confidence %:",voted_classifier.confidence(testing_set[1][0])*100)
print("Classification:",voted_classifier.classify(testing_set[2][0]),"Confidence %:",voted_classifier.confidence(testing_set[2][0])*100)
print("Classification:",voted_classifier.classify(testing_set[3][0]),"Confidence %:",voted_classifier.confidence(testing_set[3][0])*100)
print("Classification:",voted_classifier.classify(testing_set[4][0]),"Confidence %:",voted_classifier.confidence(testing_set[4][0])*100)
print("Classification:",voted_classifier.classify(testing_set[5][0]),"Confidence %:",voted_classifier.confidence(testing_set[5][0])*100)
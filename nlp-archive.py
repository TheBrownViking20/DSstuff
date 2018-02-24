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
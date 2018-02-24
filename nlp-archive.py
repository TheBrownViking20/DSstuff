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
        chunkGram = r"""Chunk: {<.*>+}]<VB.?|IN|DT|TO>+{ """
        chunkParser = nltk.RegexpParser(chunkGram)
        chunked = chunkParser.parse(tagged)
        chunked.draw()            
process_content_three()
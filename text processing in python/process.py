import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# this function processes data by removing numbers and punctuation, 
# converting each letter in lower case, removing stop words(common or unrequired words) and
# creating list and finally frequency distributions of words and prints most common 15 words
def text_process(file):
    f = open(file,'r')
    data = f.read()
    f.close()
    
    result = ' '.join(c for c in data if c.isalpha())
    result = ' '.join(filter(str.isalpha, data))  
    result = re.sub(r'[^A-Za-z]', ' ', data)
    
    res = word_tokenize(result.lower())
    
    stop_words = set(stopwords.words("english"))
    
    var = [w for w in res if not w in stop_words]
    
    var = nltk.FreqDist(var)
    print(var.most_common(15))
    return var

fantasy = text_process('fotr.txt')

scifi = text_process('scifi.txt')


# check frequencies of different words in scifi and fantasy models
print(fantasy.freq("woods"))
print(scifi.freq("woods"))

print(fantasy.freq("magic"))
print(scifi.freq("magic"))

print(fantasy.freq("project"))
print(scifi.freq("project"))

print(fantasy.freq("projectile"))
print(scifi.freq("projectile"))

print(fantasy.freq("space"))
print(scifi.freq("space"))

print(fantasy.freq("system"))
print(scifi.freq("system"))

print(fantasy.freq("meat"))
print(scifi.freq("meat"))

print(fantasy.freq("dagger"))
print(scifi.freq("dagger"))

print(fantasy.freq("staff"))
print(scifi.freq("staff"))

print(fantasy.freq("ship"))
print(scifi.freq("ship"))

print(fantasy.freq("dark"))
print(scifi.freq("dark"))

print(fantasy.freq("voyage"))
print(scifi.freq("voyage"))

print(fantasy.freq("country"))
print(scifi.freq("country"))

print(fantasy.freq("phone"))
print(scifi.freq("phone"))

print(fantasy.freq("communication"))
print(scifi.freq("communication"))
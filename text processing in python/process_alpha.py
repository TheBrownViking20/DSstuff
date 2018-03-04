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


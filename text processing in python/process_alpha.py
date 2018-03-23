import nltk
import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import os
import glob

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

# This function creates an entire file containing data of all text files for a given genre
def directory_process(genre):
    os.chdir('F:/Machine Learning/DSstuff/text processing in python/{}'.format(genre))
    files = glob.glob('F:/Machine Learning/DSstuff/text processing in python/{}/*.txt'.format(genre))
    s = ""
    for i in files:
        f = open(i,'r',encoding="latin-1")
        data = f.read()
        f.close()
        s = s + " " + data 
    text_file = open("{}.txt".format(genre), "w")
    text_file.write(s)
    text_file.close()

# Creating files for text processing
directory_process('fantasy')
directory_process('horror')
directory_process('scifi')
directory_process('mystery_or_detective')


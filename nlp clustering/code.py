# Importing libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction import text
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from nltk.tokenize import RegexpTokenizer
from nltk.stem.snowball import SnowballStemmer
%matplotlib inline

df = pd.read_csv('7282_1.csv')
df.head()

df.info()

df = df.drop_duplicates('reviews.text')

df = df.drop(df.index[[7:29,31,35]])
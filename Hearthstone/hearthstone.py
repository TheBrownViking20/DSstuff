# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Importing data
decks = pd.read_csv('data.csv')
cards = pd.read_json('refs.json')

decks.head()

decks.info()

decks.describe()

# First, we calculate the average cost to build a deck and also calculate standard deviation
decks['craft_cost'].mean()
decks['craft_cost'].std()

decks['deck_class'].value_counts().plot.bar()

for i in decks['deck_class'].unique():
    print(i,end=" -->> ")
    print(decks['craft_cost'][decks['deck_class'] == i][decks['craft_cost'] > int(decks['craft_cost'].mean())].count())

decks['deck_type'].value_counts().plot.bar()


for i in decks['deck_class'].unique():
    print(i,end=" : \n")
    decks['deck_archetype'][decks['deck_archetype'] != 'Unknown'][decks['deck_class'] == i].value_counts().plot.bar()
    plt.show()
    
for i in decks['deck_set'].unique():
    decks['deck_class'][decks['deck_set'] == i].value_counts().plot.line(label=i, figsize=(16,24))
plt.legend(bbox_to_anchor=(1.04,1), loc="upper left")
plt.show()
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

card_list = list(pd.concat([decks['card_0'],decks['card_1'],decks['card_2'],decks['card_3'],decks['card_4'],decks['card_5'],decks['card_6'],decks['card_7'],decks['card_8'],decks['card_9'],decks['card_10'],decks['card_11'],decks['card_12'],decks['card_13'],decks['card_14'],decks['card_15'],decks['card_16'],decks['card_17'],decks['card_18'],decks['card_19'],decks['card_20'],decks['card_21'],decks['card_22'],decks['card_23'],decks['card_24'],decks['card_25'],decks['card_26'],decks['card_27'],decks['card_28'],decks['card_29']]))

from collections import Counter
c = Counter(card_list)
c1000 = c.most_common(1000)
c1000 = pd.DataFrame(c1000)
c1000.columns = ['Card ID','Frequency']

for i in decks['deck_class'].unique():
    print(i,end=" : \n")
    card_list = list(pd.concat([decks['card_0'][decks['deck_class'] == i],decks['card_1'][decks['deck_class'] == i],decks['card_2'][decks['deck_class'] == i],decks['card_3'][decks['deck_class'] == i],decks['card_4'][decks['deck_class'] == i],decks['card_5'][decks['deck_class'] == i],decks['card_6'][decks['deck_class'] == i],decks['card_7'][decks['deck_class'] == i],decks['card_8'][decks['deck_class'] == i],decks['card_9'][decks['deck_class'] == i],decks['card_10'][decks['deck_class'] == i],decks['card_11'][decks['deck_class'] == i],decks['card_12'][decks['deck_class'] == i],decks['card_13'][decks['deck_class'] == i],decks['card_14'][decks['deck_class'] == i],decks['card_15'][decks['deck_class'] == i],decks['card_16'][decks['deck_class'] == i],decks['card_17'][decks['deck_class'] == i],decks['card_18'][decks['deck_class'] == i],decks['card_19'][decks['deck_class'] == i],decks['card_20'][decks['deck_class'] == i],decks['card_21'][decks['deck_class'] == i],decks['card_22'][decks['deck_class'] == i],decks['card_23'][decks['deck_class'] == i],decks['card_24'][decks['deck_class'] == i],decks['card_25'][decks['deck_class'] == i],decks['card_26'][decks['deck_class'] == i],decks['card_27'][decks['deck_class'] == i],decks['card_28'][decks['deck_class'] == i],decks['card_29'][decks['deck_class'] == i]]))
    from collections import Counter
    c = Counter(card_list)
    c50 = c.most_common(50)
    c50 = pd.DataFrame(c50)
    c50.columns = ['Card ID','Frequency']
    print(c50)
    print("\n")
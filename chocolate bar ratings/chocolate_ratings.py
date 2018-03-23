# Importing the libraries
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Importing data
data = pd.read_csv('flavors_of_cacao.csv')

data.head()

# Column names of this dataset are so long that they can hinder this analysis so first, I will change them
data.columns = ['Company','Origin','REF', 'Review_Date', 'Cocoa_Percent', 'Company_Location', 'Rating',
       'Bean_Type', 'Broad_Bean_Origin']

data['Rating'].value_counts().plot(kind='barh')
plt.xlabel('Frequency')
plt.ylabel('Ratings')
plt.show()

data['Cocoa_Percent'] = data['Cocoa_Percent'].astype(str).str[0:2]

data['Cocoa_Percent'].astype(int).mean()


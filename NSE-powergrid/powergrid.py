# Importing the libraries
import pandas as pd
from pandas import Series,DataFrame
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')

# Importing the dataset
dataset = pd.read_csv('NSE-POWERGRID.csv')

dataset['Close'].plot(legend=True,figsize(10,4))

dataset['Total Trade Quantity'].plot(legend=True)

ma_day = [10,20,50]

for ma in ma_day:
    column_name = "MA for {} days".format(str(ma))
    dataset[column_name] = Series(dataset['Close']).rolling(window=ma).mean()
    
dataset[['Close','MA for 10 days','MA for 20 days','MA for 50 days']].plot(subplots=False,figsize=(10,4))

dataset['Daily Return'] = dataset['Close'].pct_change()
dataset['Daily Return'].plot(figsize=(100,20),legend=True,linestyle='--',marker='o')

sns.distplot(dataset['Daily Return'].dropna(),bins=100,color='purple')

dataset['Daily Return'].hist(bins=100)
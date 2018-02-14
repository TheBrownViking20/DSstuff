# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('camera_dataset.csv')

dataset['Brand'] = dataset['Model'].str.split().str.get(0)

dataset['Brand'].value_counts()
dataset['Brand'].value_counts().plot.bar(figsize=(12,6))

for i in dataset.columns:
    print(i,end="")
    print(" -->> ",end="")
    print(dataset[i].isnull().sum())

for i in  dataset.drop(['Model','Brand'], axis=1).columns:
    dataset[i] = dataset[i].fillna(int(dataset[i].mean()))
    
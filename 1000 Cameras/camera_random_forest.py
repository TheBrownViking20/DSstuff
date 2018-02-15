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
    
X = dataset.iloc[:,1:12].values
y = dataset.iloc[:,12].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Fitting the SVR to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = 'rbf')
regressor.fit(X_train,y_train)

# Predicting a new result
y_pred = regressor.predict(X_test)

from sklearn.metrics import accuracy_score

print(accuracy_score(y_test, y_pred))
print(accuracy_score(y_test, y_pred, normalize=False))
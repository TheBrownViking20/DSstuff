# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Data
dataset = pd.read_csv('PUBG_Player_Statistics.csv')

X = dataset.iloc[:,2].values
y = dataset.iloc[:,9].values

X = X.reshape((87898,1))
y = y.reshape((87898,1))

for i in range(len(X)):
    X[i] = int(X[i])
    
for i in range(len(y)):
    y[i] = int(y[i])

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

# Fitting Simple Linear Regression into the Training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)

# Visualising the Training set results
plt.scatter(X_train, y_train,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('(Training set)')
plt.show()

# Visualising the Test set results
plt.scatter(X_test, y_test,color='red')
plt.plot(X_train, regressor.predict(X_train), color='blue')
plt.title('')
plt.show()

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)
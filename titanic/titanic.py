import numpy as np
import pandas as pd

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

test_buffer = test

train = train.drop(['Name','Ticket','Cabin'],axis=1)
test = test.drop(['Name','Ticket','Cabin'],axis=1)
train = train.dropna(subset=['Age','Embarked'])


X = train.iloc[:,2:].values
y = train.iloc[:,1].values
X_test = test.iloc[:,1:].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
def xencoder(X):
    labelencoder_X_1 = LabelEncoder()
    X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
    labelencoder_X_2 = LabelEncoder()
    X[:, 6] = labelencoder_X_2.fit_transform(X[:, 6])
    onehotencoder = OneHotEncoder(categorical_features = [0,6])
    X = onehotencoder.fit_transform(X).toarray()
    X = X[:,1:]
    X = X[:,:-1]
    
xencoder(X)
xencoder(X_test)

import keras
from keras.models import Sequential
from keras.layers import Dense

classifier = Sequential()

classifier.add(Dense(output_dim=5, init='uniform', activation='relu', input_dim=9))

classifier.add(Dense(output_dim=5, init='uniform', activation='relu'))

classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.fit(X, y, batch_size=1000, nb_epoch=1500)

y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)
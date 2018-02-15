# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Importing Data
dataset = pd.read_csv('F:\Machine Learning\DSstuff\Pokemon\p.csv')

X = dataset.iloc[:,5:11].values
y = dataset.iloc[:,12].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

a_name = ['Logistic Regression','KNN','SVM','Naive Bayes','Decision Trees','Random Forest']
wrong_pred_list = []

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# In the following algorithms, we will use confusion matrix which determines the count of right and wrong predictions. The elements of main or principal diagonal represent the number of correct predictions and the other diagonal represents incorrect predictions  

# -----+++++Logistic Regression+++++-----

# Fitting Logistic Regression to the Training set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)

# Predicting the test set results
y_pred = classifier.predict(X_test)

# Making the confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

wrong_pred_list.append(cm)

# -----+++++K-Nearest Neighbors (K-NN)+++++-----

# Fitting K-NN to the Training set
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

wrong_pred_list.append(cm)

# -----+++++Support Vector Machine+++++-----

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='poly',random_state=0)
classifier.fit(X_train,y_train)


# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

wrong_pred_list.append(cm)

# -----+++++Naive Bayes+++++-----

# Fitting classifier to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

wrong_pred_list.append(cm)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# -----+++++Decision Trees+++++-----

# Fitting classifier to the Training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

wrong_pred_list.append(cm)

# -----+++++Random Forest+++++-----

# Fitting classifier to the Training set
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=50,criterion='entropy',random_state=0)
classifier.fit(X_train,y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

wrong_pred_list.append(cm)

# -----+++++Choosing the best algoruithm+++++-----

output_data = pd.DataFrame({'Algorithm': a_name, 'Confusion Matrix': wrong_pred_list})

wrong_pred_sum = []
for i in output_data['Confusion Matrix']:
    wrong_pred_sum.append(i[0][1] + i[1][0])
    
output_data['Wrong Predictions'] = wrong_pred_sum

print(output_data)
print("--------------------------------------------------------------------------------------------------------------------------------------------------------------------")
print("As Logistic Regression, Support Vector Machine and Naive Bayes Algorithms have least number of errors i.e., 7 so they are the best for classifying legendary pokemon")

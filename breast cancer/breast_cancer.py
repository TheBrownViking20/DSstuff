# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid');

# Importing data
data = pd.read_csv('data.csv')

del data['Unnamed: 32']

column_list = ['radius_mean',
 'texture_mean',
 'perimeter_mean',
 'area_mean',
 'smoothness_mean',
 'compactness_mean',
 'concavity_mean',
 'concave points_mean',
 'symmetry_mean',
 'fractal_dimension_mean',
 'radius_se',
 'texture_se',
 'perimeter_se',
 'area_se',
 'smoothness_se',
 'compactness_se',
 'concavity_se',
 'concave points_se',
 'symmetry_se',
 'fractal_dimension_se',
 'radius_worst',
 'texture_worst',
 'perimeter_worst',
 'area_worst',
 'smoothness_worst',
 'compactness_worst',
 'concavity_worst',
 'concave points_worst',
 'symmetry_worst',
 'fractal_dimension_worst']

corr_1 = []
corr_2 = []
correlation = []
for i in column_list:
    for j in column_list:
        corr_1.append(i)
        corr_2.append(j)
        correlation.append(data[i].corr(data[j]))
corr_data = pd.DataFrame(
    {'Corr_1': corr_1,
     'Corr_2': corr_2,
     'Correlation': correlation
    })
    
corr_pivot_data = corr_data.pivot(values='Correlation',index='Corr_1',columns='Corr_2')

sns.heatmap(corr_pivot_data)

sns.clustermap(corr_pivot_data)

sns.pairplot(corr_pivot_data)

corr_data[corr_data['Correlation'] >= 0.85][corr_data['Correlation'] < 0.98]

X = data.iloc[:, 7].values
y = data.iloc[:, 8].values
    
# Splitting the dataset into training set and testing set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

X_train = X_train.reshape((512,1))
y_train = y_train.reshape((512,1))
X_test = X_test.reshape((57,1))
y_test = y_test.reshape((57,1))
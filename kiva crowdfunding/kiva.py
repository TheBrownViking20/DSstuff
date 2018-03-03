# Importing libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid');

# Importing data
data = pd.read_csv('kiva_loans.csv')

data.head()

data.info()

data.describe()

data['month'] =  data['date'].astype(str).str[0:7]

pivot_data = pd.pivot_table(data,values='funded_amount',index='month',columns='sector')

sns.heatmap(pivot_data)

data['funded_amount'].mean()
data['lender_count'].mean()
data['term_in_months'].mean()

print(data['funded_amount'].corr(data['lender_count']))
print(data['funded_amount'].corr(data['term_in_months']))
print(data['term_in_months'].corr(data['lender_count']))
print(data['funded_amount'].corr(data['loan_amount']))
sns.jointplot(x="funded_amount", y="loan_amount", data=data, kind='reg')

f = sns.factorplot(x="sector",data=data,kind="count",hue="repayment_interval",size=12,palette="BuGn_r")
f.set_xticklabels(rotation=90)

data['repayment_interval'].value_counts().plot(kind="pie",figsize=(12,12))

data['sector'].value_counts().plot(kind="area",figsize=(12,12))
plt.xticks(np.arange(15), tuple(data['sector'].unique()), rotation=60)
plt.show()
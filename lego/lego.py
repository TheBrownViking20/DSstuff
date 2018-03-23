# Importing the libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_style('whitegrid')
%matplotlib inline

colors = pd.read_csv('colors.csv')
inventories = pd.read_csv('inventories.csv')
inventory_parts = pd.read_csv('inventory_parts.csv')
inventory_sets = pd.read_csv('inventory_sets.csv')
part_categories = pd.read_csv('part_categories.csv')
parts = pd.read_csv('parts.csv')
sets = pd.read_csv('sets.csv')
themes = pd.read_csv('themes.csv')
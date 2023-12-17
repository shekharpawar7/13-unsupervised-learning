"""A film distribution company wants to target audience based on their 
likes and dislikes, you as a Chief Data Scientist Analyze the data and 
come up with different rules of movie list so that the business objective
 is achieved.
"""

'''
Business Objective:
    
maximize: Identify the movies like and dislike of user
 and suggest the liked movies to the user.

Minimize: minimize the dilike by user if user suggest unlike movie then user can dilike movie 

Constraints: increase the user satisfaction by providing liked movies .

'''

'''
Data dictionary:
'Sixth Sense':, 'Gladiator', 'LOTR1', 'Harry Potter1', 'Patriot',
       'LOTR2', 'Harry Potter2', 'LOTR', 'Braveheart', 'Green Mile'
      #ALL the coliumn are the movies name
'''
import pandas as pd
import seaborn as sns
df=pd.read_csv("c:/4-DataSets/my_movies.csv")
df.shape
#the dataset having a 10 row and 10 columns
v=df.describe()
df.info()
#all the data in the form of int datatype

#check any null value is thir
df.isnull().sum()
"""
Sixth Sense      0
Gladiator        0
LOTR1            0
Harry Potter1    0
Patriot          0
LOTR2            0
Harry Potter2    0
LOTR             0
Braveheart       0
Green Mile       0

#not any column having a null value
"""
# Visualization of Data
sns.boxplot(df)
#there no any outlier becoues the data in binary data

sns.pairplot(df)
# No Datapoints are corelated as the all the datapoints are in scatter form 

#3. Heatmap
corr=df.corr()
sns.heatmap(corr)
# The diagonal color of the heatmap is same as the datapoints folllow some pattern
# so we can use this data for the model building

#Normalization
#The data is numeric one so we have to perform normalization

def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(v)
df_norm

b=df_norm.describe()
# Model Building
# Association Rules
from mlxtend.frequent_patterns import apriori,association_rules

data=pd.read_csv('my_movies.csv')
data

# All the data is in properly separated form so no need to apply the encoding techique
# as it is already is in the form of numeric one

from collections import Counter
item_frequencies=Counter(data)

# Apriori algorithm
frequent_itemsets = apriori(data, min_support=0.05, use_colnames=True)

# Generate association rules
rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.5)
# This generate association rule for columns
# comprises of antescends,consequences

rules.head(20)
rules.sort_values('lift',ascending=False).head(10)

# Visualize the rules
import networkx as nx
import matplotlib.pyplot as plt

# Create directed graph from the rules
G = nx.from_pandas_edgelist(rules, 'antecedents', 'consequents')

# Draw the graph
fig, ax = plt.subplots(figsize=(14, 8))
pos = nx.spring_layout(G)
nx.draw(G, pos, with_labels=True, node_size=2500, node_color="skyblue", font_size=10, font_color="black", font_weight="bold", edge_color="gray", linewidths=1, alpha=0.7)
plt.title("Association Rules Network", fontsize=15)
plt.show()










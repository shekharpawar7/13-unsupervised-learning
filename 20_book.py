"""Kitabi Duniya, a famous book store in India, which was established 
before Independence, the growth of the company was incremental year
 by year, but due to online selling of books and wide spread Internet
 access its annual growth started to collapse, seeing sharp downfalls,
 you as a Data Scientist help this heritage book store gain its popularity
 back and increase footfall of customers and provide ways the business can
 improve exponentially, apply Association RuleAlgorithm, explain the rules,'
 and visualize the graphs for clear understanding of solution.
 
 
"""

'''
Business Objective:
    
maximize: Identify and promote book combinations that are 
frequently purchased together to increase cross-selling opportunities.

Minimize: Increase sales and revenue by promoting popular book categories 

Constraints: The business needs to address online competition.
Strategies should include both online and offline components to capture a broader market.



# the benefits/impact of the solution 
# By identifying books that are frequently purchased together,
# the bookstore can create curated bundles or recommendations, enhancing the overall 
# shopping experience for customers.
# By using this association rule we can stratergically placed the books together to encourage
# the customer to purchased more items which will help to increased the overall revenue


'''

'''
Data dictionary:

'ChildBks': Children's books category.
'YouthBks': Youth books category.
'CookBks': Cookbooks category.
'RefBks': Reference books category.
'ArtBks': Art books category.
'GeogBks': Geography books category.
'ItalCook': Italian Cookbooks category.
'ItalAtlas': Italian Atlases category.
'ItalArt': Italian Art books category.
'Florence': Possibly a location or specific book related to Florence.
'DoItYBks': Do-it-yourself books category.
'''

import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns

df=pd.read_csv('book.csv')
df
df.shape
#there are 2000 rows and 11 columns in dataset
df.columns
'''
['ChildBks', 'YouthBks', 'CookBks', 'DoItYBks', 'RefBks', 'ArtBks',
       'GeogBks', 'ItalCook', 'ItalAtlas', 'ItalArt', 'Florence']
'''
v=df.describe()
df.info()
#all the data column are int data type
'''
ChildBks     int64
YouthBks     int64
CookBks      int64
DoItYBks     int64
RefBks       int64
ArtBks       int64
GeogBks      int64
ItalCook     int64
ItalAtlas    int64
ItalArt      int64
Florence     int64'''

#check any null value or not
df.isnull().sum()
#there is not any null value
'''
ChildBks     0
YouthBks     0
CookBks      0
DoItYBks     0
RefBks       0
ArtBks       0
GeogBks      0
ItalCook     0
ItalAtlas    0
ItalArt      0
Florence     0
dtype: int64
'''
# Five Number Summary
df.DoItYBks.describe()
#there is only 0 and 1 that mean the binary data that way there is \
#max value is 1 and min value is 0
df.info()
'''
 #   Column     Non-Null Count  Dtype
---  ------     --------------  -----
 0   ChildBks   2000 non-null   int64
 1   YouthBks   2000 non-null   int64
 2   CookBks    2000 non-null   int64
 3   DoItYBks   2000 non-null   int64
 4   RefBks     2000 non-null   int64
 5   ArtBks     2000 non-null   int64
 6   GeogBks    2000 non-null   int64
 7   ItalCook   2000 non-null   int64
 8   ItalAtlas  2000 non-null   int64
 9   ItalArt    2000 non-null   int64
 10  Florence   2000 non-null   int64
dtypes: int64(11)
'''
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

data=pd.read_csv('book.csv')
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

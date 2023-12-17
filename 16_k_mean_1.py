import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from sklearn.cluster import KMeans
#let us try to understand first hoe k mean work for twon dimensional data
#for that generate random number inht range 0 to 1 and with uniform probability of 1/50
X=np.random.uniform(0,1,50)
Y=np.random.uniform(0,1,50)
#create a empty dataframe with 0 row and 2 columns
df_xy=pd.DataFrame(columns=['X','Y'])
#assign the value to columns
df_xy.X=X
df_xy.Y=Y
df_xy.plot(x="X",y="Y",kind='scatter')
model_1=KMeans(n_clusters=3).fit(df_xy)
#wirh the data x and y apply Kmean model,generate scatter plot with scale fornt =10
#cmap=plt.cm.coolwarm:cool color cobination
model_1.labels_
df_xy.plot(x="X",y="Y",c=model_1.labels_,kind="scatter",s=10,cmap=plt.cm.coolwarm)

univ1=pd.read_excel("c:/4-DataSets/University_Clustering.xlsx")
univ=univ1.drop(["State"],axis=1)

def norm_fun(i):
    x=(i - i.min() ) -(i.max()-i.min())
    return x
df_norm=norm_fun(univ.iloc[:,1:])

"""
what will be ideal cluster number,will it be 1,2 and 3
"""
TWWS=[]
k=list(range(2,8))
for i in k:
    kmean=KMeans(n_clusters=i).fit(df_norm)
    TWWS.append(kmean.inertia_)
"""Kmena inertia also know as Sum of square Error, calculates the sum of 
the distance the point. it is difffrent between the abserdved values and the 
predicticted value
"""
TWWS#k valuse increase the twws values descreses

plt.plot(k, TWWS,'ro-')
plt.xlabel("no of cluster")
plt.ylabel("total within ss")

"""
hoe to select k    valuse of k elbow curve when k change from 2 to 3
then decrese in twss id=s higher than
when k chnage from 3to 4 
when k values change from 5 to 6 decrese in twss is considerably less
hence considered k =3
"""
model=KMeans(n_clusters=3).fit(df_norm).labels_
mb=pd.Series(model)
univ['clust']=mb

univ=univ.iloc[:, [7,1,2,3,4,5,6]]

univ.iloc[:,2:8].groupby(univ.clust).mean()

univ.to_csv('kmean_unive.csv',encoding='utf-8')
import os

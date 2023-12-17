"""
Perform clustering for the airlines data to obtain optimum number of clusters. 
Draw the inferences from the clusters obtained.
Refer to EastWestAirlines.xlsx dataset.
"""
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_excel("c:/4-DataSets/EastWestAirlines.xlsx")
df.head()
df.describe()
df.columns
#we ID# col for not used remove
ewa=df.drop(['ID#'],axis=1)
ewa
#we perform normilization
def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

#applying norm function
df_norm=norm_fun(ewa.iloc[:,:])
#we not put in fun 1 st col it a nominal that way skiped
df_norm.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch
#linkage function given us hirarchical or aglomerative clustering
#ref the help for linkage
z=linkage(df_norm,method='complete',metric="euclidean")
plt.figure(figsize=(15,8));
plt.title("Hierarchchical Clustering dendrogram");
plt.xlabel("index");
plt.ylabel("Distanace");

#ref help of dendrogra 
#sch.dendrogram(z)
sch.dendrogram(z)
plt.show()
#denderogram

#applying agglomerative clustering choosing 3 as clusters from dendrigram
#whatever has been dusplayed in dendrogram is not clustering
#it is just showing number of possible cluster
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean").fit(df_norm)

#apply lables to the column
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to ewa Df
ewa['clust']=cluster_labels

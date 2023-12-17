"""2.Perform clustering for the crime data and identify the number of clusters     
       formed and draw inferences. Refer to crime_data.csv dataset."""
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("c:/4-DataSets/crime_data.csv")
df.head()
df.describe()
df.columns
#we ID# col for not used remove
crime=df.drop(['Unnamed: 0'],axis=1)
crime
#we perform normilization
def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

#applying norm function
df_norm=norm_fun(crime.iloc[:,:])
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
sch.dendrogram(z)
plt.show()
#denderogram

#AgglomerativeClustering

#applying agglomerative clustering choosing 3 as clusters from dendrigram
#whatever has been dusplayed in dendrogram is not clustering
#it is just showing number of possible cluster
from sklearn.cluster import AgglomerativeClustering
h_complete=AgglomerativeClustering(n_clusters=3,linkage='complete',metric="euclidean").fit(df_norm)

#apply lables to the column
h_complete.labels_
cluster_labels=pd.Series(h_complete.labels_)
#assign this series to ewa Df

crime['clust']=cluster_labels

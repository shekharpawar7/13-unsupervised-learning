"""
Perform clustering on mixed data. 
Convert the categorical variables to numeric by using dummies or 
label encoding and perform normalization techniques.
 The data set consists of details of customers related to their auto insurance. 
 Refer to Autoinsurance.csv dataset.
"""
 
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv("c:/4-DataSets/AutoInsurance.csv")
df.head()
df.describe()
df.columns
#we some col for not used remove
Auto=df.drop(['Customer','State','Effective To Date'],axis=1)
Auto
#we perform dummie veribels
Auto_1=pd.get_dummies(Auto)
Auto_1.columns

#removing which column having duplicate columns
Auto_1=Auto_1.drop(['Response_No','Coverage_Premium','Education_High School or Below','EmploymentStatus_Unemployed','Gender_F','Location Code_Urban','Marital Status_Single','Policy Type_Special Auto','Policy_Corporate L3','Policy_Personal L3', 'Policy_Special L3','Renew Offer Type_Offer4','Sales Channel_Web','Vehicle Class_Two-Door Car','Vehicle Size_Small'],axis=1)

#we perform normilization
def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

#applying norm function
df_norm=norm_fun(Auto_1.iloc[:,:])
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

Auto['clust']=cluster_labels 



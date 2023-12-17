import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv("c:/4-DataSets/AutoInsurance.csv")
df.head()
df.columns
Auto=df.drop(['State','Response','Coverage','Education','Effective To Date','EmploymentStatus','Gender','Location Code','Marital Status','Policy Type','Policy','Renew Offer Type','Sales Channel','Vehicle Class', 'Vehicle Size'],axis=1)

def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

df_norm=norm_fun(Auto.iloc[:,1:])
df.describe()

from scipy.cluster.hierarchy import linkage
import scipy.cluster.hierarchy as sch

z=linkage(df_norm,method="complete",metric='euclidean')

plt.figure(figsize=(15,8));
plt.title("Hierarchchical Clustering dendrogram");
plt.xlabel("index");
plt.ylabel("Distanace");

 #ref help of dendrogra 
#sch.dendrogram(z)
sch.dendrogram(z)
plt.show()
#denderogram





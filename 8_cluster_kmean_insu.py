"""3.Analyze the information given in the following 
‘Insurance Policy dataset’ to             
create clusters of persons falling in the same type. 
Refer to Insurance Dataset.csv
"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

insu=pd.read_csv("c:/4-DataSets/Insurance Dataset.csv")
#in this dataset having a insurance information include the premiumns paid
#age of the person and age , how any days to renew the insurance alos having 
#income of the person , our goal is finding the cluster number and cluster group in data set

insu.columns
#data set having 'Premiums Paid', 'Age', 'Days to Renew', 'Claims made', 'Income' column
#some column name having space remove with applying new name to it

insu.rename({'Premiums Paid':'Premiums_Paid','Days to Renew':'Days_to_Renew','Claims made':'Claims_made'},axis=1,inplace=True)

insu.shape

#dataset having 100 row and 5 columns

insu.dtypes
#all columns datatype is int only one column datatype is flaot
#we need to change datatype of Claims made
#before the change dataset check the any null value is present in data

insu.isnull().sum()
#there is no any null value in dataset

insu.describe()
#the highest income is 176500 and low is 28000
#the highest premiums paid is 29900 and mean is 12542
#the avg age of the person is 46
#the maxximum day is remaining to renew is 321 and only one is min day

#let check any outlier is present is or not
#plot the box plot to find oiutlier by using seaborn 

sns.boxplot(insu)
#2columns having outlier - Premiums_Paid,Claims made
#let remove the outlier by using winsorzition

from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Premiums_Paid'])
df1=winsor.fit_transform(insu[['Premiums_Paid']]).astype(int)
insu['Premiums_Paid']=df1#outlier can be removed

#cheack the outlier
sns.boxplot(insu)#outlier removed

#apply winsor to  Claims made
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Claims_made'])
df1=winsor.fit_transform(insu[['Claims_made']]).astype(int)
insu['Claims_made']=df1#outlier can be removed

#let check the oulier
sns.boxplot(insu)
#outlier can be removed from all data
#before the apply the clustering algoritham 

#apply normalization on data
def norm_fun(i):
    x=(i - i.min())/ (i.max()- i.min())
    return x
#apply norm funtion to dataset
df_norm=norm_fun(insu)
#data is converted 

#let apply the clustering algoritham 
from sklearn.cluster import KMeans
TWSS=[]
k=list(range(2,8))
#we donts know the k value os the cluster that way 
#we aplying all the possible number in the range of 2 to 8 in TWWS
#and plot the fig and understand the k value
for i in k:
    kmean=KMeans(n_clusters=i).fit(df_norm)
    TWSS.append(kmean.inertia_)
TWSS   
plt.plot(k,TWSS,'-ro')
plt.plot(k, TWSS,'ro-')
plt.xlabel("no of cluster")
plt.ylabel("total within ss")
#from the fig we understant the cluster value k=3
model=KMeans(n_clusters=3).fit(df_norm).labels_
mb=pd.Series(model)
#add one columns in original dataset and add new column of cluster
insu['clust']=mb







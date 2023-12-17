"""2.Perform clustering for the crime data and identify the
 number of clusters            
 formed and draw inferences. Refer to crime_data.csv dataset."""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
crime=pd.read_csv('c:/4-DataSets/crime_data.csv')
#in crime_data sets having criminal actives data of states 
#in data set having number of murder,assult,urban pop and Rapes count in state
#we use this data setv for prdicting feture cases
#find similer feture chatagore and cluster the data

crime.columns
#['Unnamed: 0', 'Murder', 'Assault', 'UrbanPop', 'Rape']
#first columns having unwanted name first we need to change the first column name

crime.rename({'Unnamed: 0':'State'},axis=1,inplace=True)
#renaming the Unnamed to state

crime.shape
#dataset having 50 row and 5 column 
#first column is nominal and 4 column is numerical

crime.describe()
#by describe function it show the five number summary
#the maximum muder number is 17.4 and min murder is 0.8
#the maximum Assault number is 337 and min is 45
#the maximum Rapes number is 46

crime.dtypes
#muder datatype is float 
#Assault datatype is int
#UrbanPop datatype is int
#Rape datatype is float
#we need to convert flaot data into int
#before cheak the any null value or not

crime.isnull().sum()
#there is no any null values

crime.Murder.astype(int)
#changing datatype as int

crime.Rape.astype(int)
#rape data also change into int
#now all dataset data in int

sns.barplot(x=crime.Murder,y=crime.State)
#Georigen having highest murder cases
#north Dakota having less muder cases

sns.barplot(x=crime.Assault,y=crime.State)
#Georgia and north carolina having most Assault case
#north Dakota and hawaii less number

sns.barplot(x=crime.Rape,y=crime.State)
#north Dakota and florida having more rape cases
#from all barplot we say north Dakota is less cases of murder,Assault but
#high in rapee cases

#let chech the any outlier is present or not
sns.boxplot(crime)
#rape column having outlier 

#remove by winsorization
from feature_engine.outliers import Winsorizer

#apply winsorization for Balance column
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Rape'])
df1=winsor.fit_transform(crime[['Rape']]).astype(int)
crime['Rape']=df1
sns.boxplot(crime.Rape)#outlier can removed
#check the outlier removed or not
sns.boxplot(crime)
#all outlier can be removed

#let perform the clutering on data
#before the performing clutering we need to apply normalization on data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

df_norm=norm_fun(crime.iloc[:,1:])
#now data are normalized

from sklearn.cluster import KMeans
TWWS=[]
#we donts know the k value os the cluster that way 
#we aplying all the possible number in the range of 2 to 8 in TWWS
#and plot the fig and understand the k value
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
#from the fig we understant the cluster value k=4
model=KMeans(n_clusters=4).fit(df_norm).labels_
mb=pd.Series(model)
#add one columns in original dataset and add new column of cluster
crime['clust']=mb

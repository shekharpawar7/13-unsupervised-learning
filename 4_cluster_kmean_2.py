"""Perform K means clustering on the airlines dataset to obtain optimum
 number of clusters. Draw the inferences from the clusters obtained.
 Refer to EastWestAirlines.xlsx dataset."""
#k-mean Algoritham
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
airline=pd.read_excel("c:/4-DataSets/EastWestAirlines.xlsx")
#the east west airlines dataset show the information of mileage of flight of
#every also bonus miles 
#our main goal is finding the similer cluster in dataset
airline.head()
airline.columns
#ID#,Balance,Qual_miles,cc1_miles,cc2_miles,cc3_miles
#Bonus_miles,Bonus_trans,Flight_miles_12mo,Flight_trans_12
#Days_since_enroll,Award?
airline.shape
#In the dataset 3999 row and 12 columns
airline.mean()
#in dataset 12 columns are the balance column mean is 73601.32758, 
#Qual mileage columns mean is 1444.11
#cc1 mileage columns having 2.05 mean
#cc2 mileage columns having 1.01 mean , also cc3 mileage 1.01 mean
#Bonus_miles columns mean is 17144.846212 and Bonus_trans mean is 11.601900
#Flight_miles_12mo mean is  460.055764,Flight_trans_12 mean is 1.373593
#Days_since_enroll  column mean is 4118.559390,and last column Award? mean is 0.370343
airline.Balance.describe()
#min balance is 0 and max is 1.704838e+06---0.0000001704
airline.isnull().sum()
#there is not any null value
airline.duplicated().sum()
#also in this dataset having zero duplicate
airline.dtypes
#all columns datatype is int

sns.boxplot(airline.iloc[:,:7])
sns.boxplot(airline.Bonus_miles)
#we will check outlier in first 6 columns
#Balance,Qual_miles,cc2_miles,cc3_miles,Bonus_miles having outliers
#we use winserizer to remove outliers


from feature_engine.outliers import Winsorizer
#apply winsorization for Balance column
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Balance'])
df1=winsor.fit_transform(airline[['Balance']]).astype(int)
airline['Balance']=df1
sns.boxplot(airline.Balance)
#balance columns outlier can be removed

#apply winsorization for Qual_miles column
#winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Qual_miles'])
#df1=winsor.fit_transform(airline[['Qual_miles']]).astype(int)
#airline['Qual_miles']=df1

#winsorization not be work becoues of low variation
#we try a masking
iqr=airline.Qual_miles.quantile(0.75)-airline.Qual_miles.quantile(0.25)
upper_limit=airline.Qual_miles.quantile(0.75) + 1.5 *iqr
lower_limit=airline.Qual_miles.quantile(0.25) - 1.5 *iqr
df_t=np.where(airline.Qual_miles > upper_limit,upper_limit,np.where(airline.Qual_miles < lower_limit ,lower_limit ,airline.Qual_miles))
airline['Qual_miles']=df_t
sns.boxplot(airline.Qual_miles)#outlier can removed

#apply masking for cc2_miles column
iqr=airline.cc2_miles.quantile(0.75)-airline.cc2_miles.quantile(0.25)
upper_limit=airline.cc2_miles.quantile(0.75) + 1.5 *iqr
lower_limit=airline.cc2_miles.quantile(0.25) - 1.5 *iqr
df_t=np.where(airline.cc2_miles > upper_limit,upper_limit,np.where(airline.cc2_miles < lower_limit ,lower_limit ,airline.cc2_miles))
airline['cc2_miles']=df_t
sns.boxplot(airline.cc2_miles)#outlier can removed

#apply masking for cc3_miles column
iqr=airline.cc3_miles.quantile(0.75)-airline.cc3_miles.quantile(0.25)
upper_limit=airline.cc3_miles.quantile(0.75) + 1.5 *iqr
lower_limit=airline.cc3_miles.quantile(0.25) - 1.5 *iqr
df_t=np.where(airline.cc3_miles > upper_limit,upper_limit,np.where(airline.cc3_miles < lower_limit ,lower_limit ,airline.cc3_miles))
airline['cc3_miles']=df_t
sns.boxplot(airline.cc3_miles)#outlier can removed

#apply winsorization for Balance column
winsor=Winsorizer(capping_method='iqr',tail='both',fold=1.5,variables=['Bonus_miles'])
df1=winsor.fit_transform(airline[['Bonus_miles']]).astype(int)
airline['Bonus_miles']=df1
sns.boxplot(airline.Bonus_miles)#outlier can removed

sns.boxplot(airline.iloc[:,6:])
#Bonus_trans,Flight_miles_12mo,Flight_trans_12 having outlier

#apply masking for Bonus_trans column
iqr=airline.Bonus_trans.quantile(0.75)-airline.Bonus_trans.quantile(0.25)
upper_limit=airline.Bonus_trans.quantile(0.75) + 1.5 *iqr
lower_limit=airline.Bonus_trans.quantile(0.25) - 1.5 *iqr
df_t=np.where(airline.Bonus_trans > upper_limit,upper_limit,np.where(airline.Bonus_trans < lower_limit ,lower_limit ,airline.Bonus_trans))
airline['Bonus_trans']=df_t
sns.boxplot(airline.Bonus_trans)#outlier can removed

#apply masking for Flight_miles_12mo column
iqr=airline.Flight_miles_12mo.quantile(0.75)-airline.Flight_miles_12mo.quantile(0.25)
upper_limit=airline.Flight_miles_12mo.quantile(0.75) + 1.5 *iqr
lower_limit=airline.Flight_miles_12mo.quantile(0.25) - 1.5 *iqr
df_t=np.where(airline.Flight_miles_12mo > upper_limit,upper_limit,np.where(airline.Flight_miles_12mo < lower_limit ,lower_limit ,airline.Flight_miles_12mo))
airline['Flight_miles_12mo']=df_t
sns.boxplot(airline.Flight_miles_12mo)#outlier can removed

#apply masking for Flight_trans_12 column
iqr=airline.Flight_trans_12.quantile(0.75)-airline.Flight_trans_12.quantile(0.25)
upper_limit=airline.Flight_trans_12.quantile(0.75) + 1.5 *iqr
lower_limit=airline.Flight_trans_12.quantile(0.25) - 1.5 *iqr
df_t=np.where(airline.Flight_trans_12 > upper_limit,upper_limit,np.where(airline.Flight_trans_12 < lower_limit ,lower_limit ,airline.Flight_trans_12))
airline['Flight_trans_12']=df_t
sns.boxplot(airline.Flight_trans_12)#outlier can removed

sns.boxplot(airline)
#removing all outlier from datasets

airline.isnull().sum()#not any null value

#performing normlization on numeric data
def norm_fun(i):
    x=(i - i.min() ) -(i.max()-i.min())
    return x

df_norm=norm_fun(airline.iloc[:,1:])

from sklearn.cluster import KMeans
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
model=KMeans(n_clusters=3).fit(df_norm).labels_
mb=pd.Series(model)
airline['clust']=mb






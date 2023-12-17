"""A pharmaceuticals manufacturing company is conducting a study on a
 new medicine to treat heart diseases. The company has gathered data
 from its secondary sources and would like you to provide high level 
 analytical insights on the data. Its aim is to segregate patients 
 depending on their age group and other factors given in the data.
 Perform PCA and clustering algorithms on the dataset and check if the
 clusters formed before and after PCA are the same and provide a brief
 report on your model. You can also explore more ways to improve your 
 model. """
#the our  mean aim is the perform the clustering alogritham on the dataset and 
#form the group base on the data feture in dataset
#then we perform the PCA alogrithom to the reduce the feture in data set
#it help to improve the perfoemce of the model
#it help to pharmaceuticals company to study on the new mwdicine in hesrt disease
 
#data dictionary
#age-the age column show the age of patient
#sex-the sex column shoe the patient male or female
#cp-it is Constrictive pericarditis score in test
#trestbps - The person's resting blood pressure (mm Hg on admission to the hospital)
#chol-Cholesterol persentage in the blood
#fbs-The person's fasting blood sugar of patient
#restecg-Resting electrocardiographic measurement 
#thalach-The person's maximum heart rate achieved
#exang- the exercise induced angina which is recorded as 1 if there is pain and 0 if there is no pain
#oldpeak - ST depression caused by activity in comparison to rest.
#slope- a sensitive and specific marker of transient myocardial ischemia
#ca-The coronary artery calcium (CAC) score measures the amount
#thal-Thalassemia (thal-uh-SEE-me-uh) is an inherited blood disorder that causes your body to have less hemoglobin than normal.
#target-is group in the dataset

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#read dataset ino the pharm varible
pharm=pd.read_csv("c:/4-DataSets/heart disease.csv")
pharm.shape
#there are a 303 rows and 14 columns

pharm.columns
#there 14 columns-'age', 'sex', 'cp', 'trestbps', 'chol',
#bs', 'restecg', 'thalach','exang', 'oldpeak', 'slope', 'ca', 
#'thal', 'target'

pharm.describe()
#the higest age of patient is 77 age old and samllest age is 29
#cp is the max 3 and 0 is low same patient having 0 cp
#the slope maximum is the 2 and lowest is 0
#the chaol is max is 564 and lowest is 126

pharm.info()
#in this data set y=the oldpeak is only column is float and 
#all column are in the int datatype 
#there is no any null value

sns.boxplot(pharm)
#trestbps ,chol,fbs,thalsch,oldpeak,ca,thal column havinh outlier 
#remove the outlier by using a masking ,treming or winsolizer

#remove the outlier of trestbps by winsoizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['trestbps'])
df2=winsor.fit_transform(pharm[['trestbps']]).astype(int)
pharm['trestbps']=df2#outlier can be removed
sns.boxplot(pharm)

#remove the outlier of chol by winsoizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['chol'])
df2=winsor.fit_transform(pharm[['chol']]).astype(int)
pharm['chol']=df2#outlier can be removed
sns.boxplot(pharm)

#apply masking to  Ash
import numpy as np
IQR=pharm.fbs.quantile(0.75) - pharm.fbs.quantile(0.25)
IQR
lower_limit=pharm.fbs.quantile(0.25) - 1.5 * IQR
upper_limit=pharm.fbs.quantile(0.75) + 1.5 * IQR
outlier=np.where(pharm.fbs > upper_limit,upper_limit,np.where(pharm.fbs < lower_limit ,lower_limit,pharm.fbs))
pharm.fbs=outlier
#let check removed or not
sns.boxplot(pharm.fbs)

#remove the outlier of thalach by winsoizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['thalach'])
df2=winsor.fit_transform(pharm[['thalach']]).astype(int)
pharm['thalach']=df2#outlier can be removed
sns.boxplot(pharm)


#remove the outlier of oldpeak by winsoizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['oldpeak'])
df2=winsor.fit_transform(pharm[['oldpeak']]).astype(int)
pharm['oldpeak']=df2#outlier can be removed
sns.boxplot(pharm)

#remove the outlier of ca by winsoizer
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['ca'])
df2=winsor.fit_transform(pharm[['ca']]).astype(int)
pharm['ca']=df2#outlier can be removed
sns.boxplot(pharm)

#apply masking to thal;
import numpy as np
IQR=pharm.thal.quantile(0.75) - pharm.thal.quantile(0.25)
IQR
lower_limit=pharm.thal.quantile(0.25) - 1.5 * IQR
upper_limit=pharm.thal.quantile(0.75) + 1.5 * IQR
outlier=np.where(pharm.thal > upper_limit,upper_limit,np.where(pharm.thal < lower_limit ,lower_limit,pharm.thal))
pharm.thal=outlier
#let check removed or not
sns.boxplot(pharm)

#all outlier are  removed
#the data having some ordinal value let convert it into dummie value
df1=pd.get_dummies(pharm)
#apply normalization on data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#apply norm funtion to dataset
df_norm=norm_fun(df1)

#data is converted
df_norm.drop(['fbs'],axis=1,inplace=True)


#let apply the clustering algoritham 
from sklearn.cluster import KMeans
TWSS=[]
k=list(range(2,8))
k
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
pharm['clust']=mb
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()
X_sclaer=scaler.fit_transform(df_norm)
X_sclaer


pca=PCA(n_components=6)
pca_values=pca.fit_transform(X_sclaer)

pca_values.shape
#178 and 6 columns

#convert new feture into datafream
pca_data=pd.DataFrame(pca_values)
#give new name to datafream feture
pca_data.columns='comp0','comp1','comp2','comp3','comp4','comp5'
pca_data
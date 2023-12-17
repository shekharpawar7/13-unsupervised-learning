"""3.Perform clustering analysis on the telecom data set. 
The data is a mixture of both categorical and numerical data.
 It consists of the number of customers who churn out. Derive insights and 
 get possible information on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset."""
 
import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_excel("c:/4-DataSets/Telco_customer_churn.xlsx")
df.head()
df.describe()
df.columns
#we some col for not used remove
Telco=df.drop(['Customer ID','Quarter','Offer','Offer','Contract','Payment Method'],axis=1)
Telco
#we perform dummie veribels
Telco_1=pd.get_dummies(Telco)
Telco_1.columns

#removing which column having duplicate columns
Telco_1=Telco_1.drop(['Phone Service_No','Referred a Friend_No','Multiple Lines_No','Internet Service_No','Online Security_No','Online Backup_No','Device Protection Plan_No','Premium Tech Support_No','Streaming TV_No','Streaming Movies_No', 'Streaming Music_No','Unlimited Data_No','Paperless Billing_No'],axis=1)

#rename the column
Telco_1=Telco_1.rename({'Phone Service_Yes':'Phone Service','Referred a Friend_Yes':'Referred a Friend','Multiple Lines_Yes':"Multiple Lines",'Internet Service_Yes':'Internet Service','Online Security_Yes':'Online Security','Online Backup_Yes':'Online Backup','Device Protection Plan_Yes':'Device Protection Plan','Premium Tech Support_Yes':'Premium Tech Support','Streaming TV_Yes':'Streaming TV','Streaming Movies_Yes':'Streaming Movies', 'Streaming Music_Yes':"Streaming Music",'Unlimited Data_Yes':"Unlimited Data",'Paperless Billing_Yes':'Paperless Billing'},axis=1)

#we perform normilization
def norm_fun(i):
    x=(i-i.min()) / (i.max()-i.min())
    return x

#applying norm function
df_norm=norm_fun(Telco_1.iloc[:,:])
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

Telco['clust']=cluster_labels 



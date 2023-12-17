"""5.Perform clustering on mixed data.
 Convert the categorical variables to numeric by using dummies or
 label encoding and perform normalization techniques.
 The dataset has the details of customers related to their auto insurance. 
 Refer to Autoinsurance.csv dataset."""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
df=pd.read_csv("c:/4-datasets/AutoInsurance.csv")
#in this dataset thir is a vehicle insurance infromation of customer 
#also mentions the customer infromation and vehicle infromation
#our main goal is the indintify the feture realation of reach other by 
#performing k mean clustering algoritham
df.shape 
#there is a 9134 rows and 24 columns
df.columns
"""['Customer', 'State', 'Customer Lifetime Value', 'Response', 'Coverage',
       'Education', 'Effective To Date', 'EmploymentStatus', 'Gender',
       'Income', 'Location Code', 'Marital Status', 'Monthly Premium Auto',
       'Months Since Last Claim', 'Months Since Policy Inception',
       'Number of Open Complaints', 'Number of Policies', 'Policy Type',
       'Policy', 'Renew Offer Type', 'Sales Channel', 'Total Claim Amount',
       'Vehicle Class', 'Vehicle Size']"""
#let perfrom the eda on data
df.describe()
#maximum Customer Lifetime Value is a 83325
#highest Total Claim Amount is 2893
#first  customer and state column can drop

#rename the columns which having a space
df.rename({'Customer Lifetime Value':'Customer_Lifetime_Value',
           'Effective To Date':'Effective_To_Date','Location Code':'Location_Code',
           'Marital_Status':'Marital_Status','Monthly Premium Auto':'Monthly_Premium_Auto',
           'Months Since Last Claim':'Months_Since_Last_Claim','Months Since Policy Inception':'Months_Since_Policy_Inception',
           'Number of Open Complaints':'Number_of_Open_Complaints','Number of Policies':'Number_of_Policies',
           'Policy Type':'Policy_Type','Renew Offer Type':'Renew_Offer_Type','Sales Channel':'Sales_Channel','Total Claim Amount':'Total_Claim_Amount',
           'Vehicle Class':'Vehicle_Class','Vehicle Size':'Vehicle_Size'},axis=1,inplace=True)
df.drop(['Customer','State'],axis=1,inplace=True)
#first two column can droped
sns.barplot(df) 
#bar can plot only numeric data
#check the outlier is present or not
sns.boxplot(df)
#Customer Lifetime Value,Monthly Premium Auto,
#Number of Open Complaints,Number of Policies,
#Total Claim Amount having outlier
#let remove the outlier by masking and winsorization

#apply winsor to  Avg_Monthly_GB_Download
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Customer_Lifetime_Value'])
df1=winsor.fit_transform(df[['Customer_Lifetime_Value']]).astype(int)
df['Customer_Lifetime_Value']=df1#outlier can be removed
#let check removed or not
sns.boxplot(df)

#apply masking to  Monthly_Premium_Auto
import numpy as np
IQR=df.Monthly_Premium_Auto.quantile(0.75) - df.Monthly_Premium_Auto.quantile(0.25)
IQR
lower_limit=df.Monthly_Premium_Auto.quantile(0.25) - 1.5 * IQR
upper_limit=df.Monthly_Premium_Auto.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Monthly_Premium_Auto > upper_limit,upper_limit,np.where(df.Monthly_Premium_Auto < lower_limit ,lower_limit,df.Monthly_Premium_Auto))
df.Monthly_Premium_Auto=outlier
#let check removed or not
sns.boxplot(df)

#apply masking to  Number_of_Open_Complaints
import numpy as np
IQR=df.Number_of_Open_Complaints.quantile(0.75) - df.Number_of_Open_Complaints.quantile(0.25)
IQR
lower_limit=df.Number_of_Open_Complaints.quantile(0.25) - 1.5 * IQR
upper_limit=df.Number_of_Open_Complaints.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Number_of_Open_Complaints > upper_limit,upper_limit,np.where(df.Number_of_Open_Complaints < lower_limit ,lower_limit,df.Number_of_Open_Complaints))
df.Number_of_Open_Complaints=outlier
#let check removed or not
sns.boxplot(df)

#apply masking to  Number_of_Open_Complaints
import numpy as np
IQR=df.Number_of_Policies.quantile(0.75) - df.Number_of_Policies.quantile(0.25)
IQR
lower_limit=df.Number_of_Policies.quantile(0.25) - 1.5 * IQR
upper_limit=df.Number_of_Policies.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Number_of_Policies > upper_limit,upper_limit,np.where(df.Number_of_Policies < lower_limit ,lower_limit,df.Number_of_Policies))
df.Number_of_Policies=outlier
#let check removed or not
sns.boxplot(df)

#apply masking to  Total_Claim_Amount
import numpy as np
IQR=df.Total_Claim_Amount.quantile(0.75) - df.Total_Claim_Amount.quantile(0.25)
IQR
lower_limit=df.Total_Claim_Amount.quantile(0.25) - 1.5 * IQR
upper_limit=df.Total_Claim_Amount.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Total_Claim_Amount > upper_limit,upper_limit,np.where(df.Total_Claim_Amount < lower_limit ,lower_limit,df.Total_Claim_Amount))
df.Total_Claim_Amount=outlier
#let check removed or not
sns.boxplot(df)
#the data having some ordinal value let convert it into dummie value
ddf=pd.get_dummies(df)
#apply normalization on data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#apply norm funtion to dataset
df_norm=norm_fun(ddf)
#data is converted
df_norm.drop(['Number_of_Open_Complaints'],axis=1,inplace=True)

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
df['clust']=mb












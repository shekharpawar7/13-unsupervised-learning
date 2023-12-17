"""4.Perform clustering analysis on the telecom dataset. 
The data is a mixture of both categorical and numerical data.
 It consists of the number of customers who churn. 
 Derive insights and get possible information on factors that may affect the churn decision.
 Refer to Telco_customer_churn.xlsx dataset"""
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
tel=pd.read_excel("c:/4-DataSets/Telco_customer_churn.xlsx")
#In this data set Telcom company infrmation like any subcribtion anf finacial 
#data payment having customer id and Quarter ,friend refre or not ,
#any offer are active or not
#our main goal is finading feture realtion by clustering method
tel.columns
"""['Customer ID', 'Count', 'Quarter', 'Referred a Friend',
      'Number of Referrals', 'Tenure in Months', 'Offer', 'Phone Service',
       'Avg Monthly Long Distance Charges', 'Multiple Lines',
       'Internet Service', 'Internet Type', 'Avg Monthly GB Download',
       'Online Security', 'Online Backup', 'Device Protection Plan',
       'Premium Tech Support', 'Streaming TV', 'Streaming Movies',
       'Streaming Music', 'Unlimited Data', 'Contract', 'Paperless Billing',
       'Payment Method', 'Monthly Charge', 'Total Charges', 'Total Refunds',
       'Total Extra Data Charges', 'Total Long Distance Charges',
       'Total Revenue']"""
tel.shape
#there is 7043 row and 30 columns
#there is many columns are not be usefull that way we drop it
#it is a nominal data
tel.drop(['Customer ID'],axis=1,inplace=True)
#In this dataset are ordinal data and nominaal data that we need to perform dummie varible
df=pd.get_dummies(tel)
#we get the dummie data from .get_dummies function
#but becoues of performance time we need to reduce the number of column
#perform on dummie values (n-1)
df.shape
#column can increse 30 to 54
df.columns

#drop the dummies values (n-1)
df.drop(['Offer_Offer E','Phone Service_No','Multiple Lines_No','Internet Service_No','Internet Type_None','Online Security_No','Online Backup_No','Device Protection Plan_No','Premium Tech Support_No','Streaming TV_No','Streaming Movies_No','Streaming Music_No','Unlimited Data_No','Contract_Two Year','Paperless Billing_No','Payment Method_Mailed Check'],axis=1,inplace=True)
#we reduce the 54 to 37 columns

#naming of column is not a good need to change the column name
df.columns
df.rename({'Number of Referrals':'Number_of_Referrals','Tenure in Months':'Tenure_in_Months',
           'Avg Monthly Long Distance Charges':'Avg_Monthly_Long_Distance_Charges',
           'Avg Monthly GB Download':'Avg_Monthly_GB_Download','Monthly Charge':'Monthly_Charge',
           'Total Charges':'Total_Charges','Total Refunds':'Total_Refunds',
           'Total Extra Data Charges':'Total_Extra_Data_Charges',
           'Total Long Distance Charges':'Total_Long_Distance_Charges',
           'Total Revenue':'Total_Revenue','Referred a Friend_No':'Referred_a_Friend_No',
           'Offer_Offer A':'Offer_Offer_A','Offer_Offer B':'Offer_Offer_B','Offer_Offer C':'Offer_Offer_C',
           'Offer_Offer D':'Offer_Offer_D','Phone Service_Yes':'Phone_Service','Multiple Lines_Yes':'Multiple_Lines',
           'Internet Service_Yes':'Internet_Service','Internet Type_Cable':'Internet_Type_Cable',
           'Internet Type_DSL':'Internet_Type_DSL','Internet Type_Fiber Optic':'Internet Type_Fiber_Optic',
           'Online Security_Yes':'Online_Security','Online Backup_Yes':'Online_Backup','Device Protection Plan_Yes':'Device_Protection_Plan',
           'Premium Tech Support_Yes':'Premium_tech_Support','Streaming TV_Yes':'Streaming_TV',
           'Streaming Movies_Yes':'Streaming_Movies','Streaming Music_Yes':'Streaming_Music',
           'Unlimited Data_Yes':'Unlimited_Data','Contract_Month-to-Month':'Contract_Month_to_Month',
           'Contract_One Year':'Contract_One_Year','Paperless Billing_Yes':'Paperless_Billing','Payment Method_Bank Withdrawal':'Payment_Method_Bank_Withdrawal',
           'Payment Method_Credit Card':'Payment_Method_Credit_Card'},axis=1,inplace=True)

df.shape
#now click outlier is present or not
sns.boxplot(df.iloc[:,:10])
#number of referranls having a outlier
#apply winsor to  Number_of_Referrals
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Number_of_Referrals'])
df1=winsor.fit_transform(df[['Number_of_Referrals']]).astype(int)
df['Number_of_Referrals']=df1#outlier can be removed
#let check removed or not
sns.boxplot(df.iloc[:,:10])

#apply winsor to  Avg_Monthly_GB_Download
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Avg_Monthly_GB_Download'])
df1=winsor.fit_transform(df[['Avg_Monthly_GB_Download']]).astype(int)
df['Avg_Monthly_GB_Download']=df1#outlier can be removed
#let check removed or not
sns.boxplot(df.iloc[:,:10])

#apply masking to  Total_Refunds
import numpy as np
IQR=df.Total_Refunds.quantile(0.75) - df.Total_Refunds.quantile(0.25)
IQR
lower_limit=df.Total_Refunds.quantile(0.25) - 1.5 * IQR
upper_limit=df.Total_Refunds.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Total_Refunds > upper_limit,upper_limit,np.where(df.Total_Refunds < lower_limit ,lower_limit,df.Total_Refunds))
df.Total_Refunds=outlier
#let check removed or not
sns.boxplot(df.iloc[:,:10])

#apply masking to  Total_Refunds
import numpy as np
IQR=df.Total_Extra_Data_Charges.quantile(0.75) - df.Total_Extra_Data_Charges.quantile(0.25)
IQR
lower_limit=df.Total_Extra_Data_Charges.quantile(0.25) - 1.5 * IQR
upper_limit=df.Total_Extra_Data_Charges.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Total_Extra_Data_Charges > upper_limit,upper_limit,np.where(df.Total_Extra_Data_Charges < lower_limit ,lower_limit,df.Total_Extra_Data_Charges))
df.Total_Extra_Data_Charges=outlier
#let check removed or not
sns.boxplot(df.iloc[:,:10])

#apply winsor to  Avg_Monthly_GB_Download
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Total_Long_Distance_Charges'])
df1=winsor.fit_transform(df[['Total_Long_Distance_Charges']]).astype(int)
df['Total_Long_Distance_Charges']=df1#outlier can be removed
#let check removed or not
sns.boxplot(df.iloc[:,:10])


#let check next 10
sns.boxplot(df.iloc[:,9:20])
#apply winsor to  Total_Revenue
from feature_engine.outliers import Winsorizer
winsor=Winsorizer(capping_method='iqr', tail='both',fold=1.4,variables=['Total_Revenue'])
df1=winsor.fit_transform(df[['Total_Revenue']]).astype(int)
df['Total_Revenue']=df1#outlier can be removed
#let check removed or not
sns.boxplot(df.iloc[:,:10])

#apply masking to  Offer_Offer_A
import numpy as np
IQR=df.Offer_Offer_A.quantile(0.75) - df.Offer_Offer_A.quantile(0.25)
IQR
lower_limit=df.Offer_Offer_A.quantile(0.25) - 1.5 * IQR
upper_limit=df.Offer_Offer_A.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Offer_Offer_A > upper_limit,upper_limit,np.where(df.Offer_Offer_A < lower_limit ,lower_limit,df.Offer_Offer_A))
df.Offer_Offer_A=outlier
#let check removed or not
sns.boxplot(df.iloc[:,9:20])

#apply masking to  Offer_Offer_B
import numpy as np
IQR=df.Offer_Offer_B.quantile(0.75) - df.Offer_Offer_B.quantile(0.25)
IQR
lower_limit=df.Offer_Offer_B.quantile(0.25) - 1.5 * IQR
upper_limit=df.Offer_Offer_B.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Offer_Offer_B > upper_limit,upper_limit,np.where(df.Offer_Offer_B < lower_limit ,lower_limit,df.Offer_Offer_B))
df.Offer_Offer_B=outlier
#let check removed or not
sns.boxplot(df.iloc[:,9:20])

#apply masking to  Offer_Offer_C
import numpy as np
IQR=df.Offer_Offer_C.quantile(0.75) - df.Offer_Offer_C.quantile(0.25)
IQR
lower_limit=df.Offer_Offer_C.quantile(0.25) - 1.5 * IQR
upper_limit=df.Offer_Offer_C.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Offer_Offer_C > upper_limit,upper_limit,np.where(df.Offer_Offer_C < lower_limit ,lower_limit,df.Offer_Offer_C))
df.Offer_Offer_C=outlier
#let check removed or not
sns.boxplot(df.iloc[:,9:20])

#apply masking to  Offer_Offer_D
import numpy as np
IQR=df.Offer_Offer_D.quantile(0.75) - df.Offer_Offer_D.quantile(0.25)
IQR
lower_limit=df.Offer_Offer_D.quantile(0.25) - 1.5 * IQR
upper_limit=df.Offer_Offer_D.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Offer_Offer_D > upper_limit,upper_limit,np.where(df.Offer_Offer_D < lower_limit ,lower_limit,df.Offer_Offer_D))
df.Offer_Offer_D=outlier
#let check removed or not
sns.boxplot(df.iloc[:,9:20])

#apply masking to  Phone_Service
import numpy as np
IQR=df.Phone_Service.quantile(0.75) - df.Phone_Service.quantile(0.25)
IQR
lower_limit=df.Phone_Service.quantile(0.25) - 1.5 * IQR
upper_limit=df.Phone_Service.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Phone_Service > upper_limit,upper_limit,np.where(df.Phone_Service < lower_limit ,lower_limit,df.Phone_Service))
df.Phone_Service=outlier
#let check removed or not
sns.boxplot(df.iloc[:,9:20])

#let chack next 10
sns.boxplot(df.iloc[:,19:30])
#apply masking to  Internet_Service
import numpy as np
IQR=df.Internet_Service.quantile(0.75) - df.Internet_Service.quantile(0.25)
IQR
lower_limit=df.Internet_Service.quantile(0.25) - 1.5 * IQR
upper_limit=df.Internet_Service.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Internet_Service > upper_limit,upper_limit,np.where(df.Internet_Service < lower_limit ,lower_limit,df.Internet_Service))
df.Internet_Service=outlier
#let check removed or not
sns.boxplot(df.iloc[:,19:30])

#apply masking to  Internet_Type_Cable
import numpy as np
IQR=df.Internet_Type_Cable.quantile(0.75) - df.Internet_Type_Cable.quantile(0.25)
IQR
lower_limit=df.Internet_Type_Cable.quantile(0.25) - 1.5 * IQR
upper_limit=df.Internet_Type_Cable.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Internet_Type_Cable > upper_limit,upper_limit,np.where(df.Internet_Type_Cable < lower_limit ,lower_limit,df.Internet_Type_Cable))
df.Internet_Type_Cable=outlier
#let check removed or not
sns.boxplot(df.iloc[:,19:30])

#apply masking to  Internet_Type_DSL
import numpy as np
IQR=df.Internet_Type_DSL.quantile(0.75) - df.Internet_Type_DSL.quantile(0.25)
IQR
lower_limit=df.Internet_Type_DSL.quantile(0.25) - 1.5 * IQR
upper_limit=df.Internet_Type_DSL.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Internet_Type_DSL > upper_limit,upper_limit,np.where(df.Internet_Type_DSL < lower_limit ,lower_limit,df.Internet_Type_DSL))
df.Internet_Type_DSL=outlier
#let check removed or not
sns.boxplot(df.iloc[:,19:30])

#check next column
sns.boxplot(df.iloc[:,29:])
#apply masking to  Contract_One_Year
import numpy as np
IQR=df.Contract_One_Year.quantile(0.75) - df.Contract_One_Year.quantile(0.25)
IQR
lower_limit=df.Contract_One_Year.quantile(0.25) - 1.5 * IQR
upper_limit=df.Contract_One_Year.quantile(0.75) + 1.5 * IQR
outlier=np.where(df.Contract_One_Year > upper_limit,upper_limit,np.where(df.Contract_One_Year < lower_limit ,lower_limit,df.Contract_One_Year))
df.Contract_One_Year=outlier
#let check removed or not
sns.boxplot(df.iloc[:,29:])
#all outlier can removed from data
#apply normalization on data
def norm_fun(i):
    x=(i-i.min())/(i.max()-i.min())
    return x

#apply norm funtion to dataset
df_norm=norm_fun(df)
#data is converted
df_norm.isnull().sum() 
df_norm.drop(['Count','Total_Refunds','Total_Extra_Data_Charges','Quarter_Q3','Offer_Offer_A','Offer_Offer_B','Offer_Offer_C','Offer_Offer_D','Phone_Service','Internet_Service','Internet_Type_Cable','Internet_Type_DSL','Contract_One_Year'],axis=1,inplace=True)
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
tel['clust']=mb












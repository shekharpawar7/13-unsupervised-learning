#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import eig


# In[4]:


#defining a simple data
marks=np.array([[3,4],[2,8],[6,9]])
print(marks)


# In[7]:


mark_df=pd.DataFrame(marks,columns=['Physics','math'])


# In[8]:


mark_df


# # step 1

# In[9]:


plt.scatter(mark_df.Physics,mark_df.math)


# # step 2

# In[11]:


#make data mean centric
Meanbycolumn=np.mean(marks.T,axis=1)
print(Meanbycolumn)


# In[12]:


Scaled_data=marks-Meanbycolumn


# In[13]:


marks.T


# In[14]:


Scaled_data


# # step 3

# In[16]:


#find coveariace matrix of above scaled data
Cov_mat=np.cov(Scaled_data.T)
Cov_mat


# # step 4

# In[20]:


#finding corresponding eigen value and eign vector of above cavariance matrix
Eval,Evac=eig(Cov_mat)
print(Eval)
print(Evac)


# # step 5

# In[22]:


#Get Original Data Projection to principal components as new axis
Projected_data=Evac.T.dot(Scaled_data.T)
print(Projected_data)


# In[25]:


from sklearn.decomposition import PCA
pca=PCA(n_components=2)
pca.fit_transform(marks)


# In[26]:


uni1=pd.read_excel('c:/4-DataSets/University_Clustering.xlsx')


# In[27]:


uni1.describe()


# In[28]:


uni1.info()


# In[30]:


uni=uni1.drop(['State'],axis=1)


# In[32]:


from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from sklearn.preprocessing import scale


# In[33]:


#considering only numerical data
uni.data=uni.iloc[:,1:]


# In[34]:


#Normalizling the numerical data
uni_norm=scale(uni.data)


# In[35]:


pca=PCA(n_components=6)
pca_values=pca.fit_transform(uni_norm)


# In[36]:


#the amount of varience that each PCA explains is 
var=pca.explained_variance_ratio_
var


# In[37]:


#cumulative varience
var1=np.cumsum(np.round(var,decimals=4)*100)
var1


# In[38]:


#variance plot for PCA compoents obtained
plt.plot(var1,color='r')


# In[39]:


#PCA scorees
pca_values


# In[40]:


pca_data=pd.DataFrame(pca_values)
pca_data.columns='comp0','comp1','comp2','comp3','comp4','comp5'
final=pd.concat([uni.Univ,pca_data.iloc[:,0:3]],axis=1)


# In[41]:


ax=final.plot(x='comp0',y='comp1',kind='scatter')


# In[42]:


final[['comp0','comp1','Univ']].apply(lambda x: ax.text(*x),axis=1)


# In[ ]:





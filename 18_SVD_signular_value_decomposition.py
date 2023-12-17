import pandas as pd
import numpy as np

A=np.array([[1,0,0,0,2],[0,0,3,0,0],[0,0,0,0,0],[0,4,0,0,0]])
print(A)

#SVD
from scipy.linalg import svd

U,D,Vt=svd(A)
print(U)
print(D)
print(Vt)
print(np.diag(D))


#apply svd on dataset
df=pd.read_excel("c:/4-DataSets/University_Clustering.xlsx")
data=df.iloc[:,2:]
from sklearn.decomposition import TruncatedSVD

svd=TruncatedSVD(n_components=3)
svd.fit(data)
result=pd.DataFrame(svd.transform(data))
result.head()
result.columns='pc0','pc1','pc2'

#scatter plot
import  matplotlib.pyplot as plt
plt.scatter(x=result.pc0, y=result.pc1)





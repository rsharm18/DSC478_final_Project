#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.cluster import KMeans

get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


def userDataClassification(user_data, class_column, columns):
    from sklearn.model_selection import train_test_split
    class_data = user_data[class_column]
    mod_user_data = user_data[columns]
    mod_user_data = pd.get_dummies(mod_user_data)
    
    vs_train, vs_test, vs_target_train, vs_target_test = train_test_split(mod_user_data, class_data, test_size=0.2, random_state=33)
    
    return vs_train, vs_test, vs_target_train, vs_target_test


# In[6]:


def getClusterData(K, train_data):
    kmean = KMeans(n_clusters=K);
    
    clusters = kmean.fit_transform(train_data)
    
    pd.options.display.float_format='{:,.2f}'.format

    centroids = pd.DataFrame(kmean.cluster_centers_, columns=train_data.columns)

    return kmean, clusters, centroids
    


# In[ ]:





# In[ ]:





# In[1]:


## Distance Calculation Methods
def euclidSim(inA,inB):
    return 1.0 / (1.0 + la.norm(inA - inB))

def pearsonSim(inA,inB):
    if len(inA) < 3 : return 1.0
    return 0.5 + 0.5 * corrcoef(inA, inB, rowvar = 0)[0][1]

def cosineSim(inA,inB):
    num = float(inA.T * inB)
    denom = la.norm(inA)*la.norm(inB)
    return 0.5 + 0.5 * (num / denom)


# In[2]:


def standEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    simTotal = 0.0; ratSimTotal = 0.0
    data=mat(dataMat)
    for j in range(n):
        userRating = data[user,j]
        if userRating == 0: continue
        overLap = nonzero(logical_and(data[:,item]>0, data[:,j]>0))[0]
        if len(overLap) == 0:
            similarity = 0
        else:
            similarity = simMeas(data[overLap,item], data[overLap,j])
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


# In[3]:


from numpy import linalg as la

def svdEst(dataMat, user, simMeas, item):
    n = shape(dataMat)[1]
    k = 4 #number of dimension for SVD
    simTotal = 0.0; ratSimTotal = 0.0
    data=mat(dataMat)
    U,Sigma,VT = la.svd(data)
    Sig_k = mat(eye(k)*Sigma[:k]) #arrange Sig_k into a diagonal matrix
    xformedItems = data.T * U[:,:k] * Sig_k.I  #create transformed items
    for j in range(n):
        userRating = data[user,j]
        if userRating == 0 or j==item: continue
        similarity = simMeas(xformedItems[item,:].T, xformedItems[j,:].T)
        #print('the %d and %d similarity is: %f' % (item, j, similarity))
        simTotal += similarity
        ratSimTotal += similarity * userRating
    if simTotal == 0: return 0
    else: return ratSimTotal/simTotal


# In[ ]:





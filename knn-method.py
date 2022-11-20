#!/usr/bin/env python
# coding: utf-8

# ## K Nearest Neighbors Project - Solution
# Welcome to the KNN Project! This will be a simple project very similar to the lecture, except you'll be given another data set. Go ahead and just follow the directions below.
# 
# ### Import Libraries
# #### Import pandas,seaborn, and the usual libraries

# In[33]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# ## Get the Data
# ** Read the 'KNN_Project_Data csv file into a dataframe **

# In[34]:


df = pd.read_csv('KNN_Project_Data.txt')


# In[35]:


df.head() 


# ## EDA
# Since this data is artificial, we'll just do a large pairplot with seaborn.
# 
# Use seaborn on the dataframe to create a pairplot with the hue indicated by the TARGET CLASS column.

# In[36]:


# THIS IS GOING TO BE A VERY LARGE PLOT
sns.pairplot(df,hue='TARGET CLASS',palette='coolwarm')


# ## Standardize the Variables
# Time to standardize the variables.
# 
# ** Import StandardScaler from Scikit learn.**

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


#** Create a StandardScaler() object called scaler.**
scaler = StandardScaler()


# In[39]:


#** Fit scaler to the features.**
scaler.fit(df.drop('TARGET CLASS',axis=1))


# In[40]:


#Use the .transform() method to transform the features to a scaled version.
scaled_features = scaler.transform(df.drop('TARGET CLASS',axis=1))


# In[41]:


scaled_features


# In[42]:


df.columns


# In[43]:


df.columns[:-1]


# Convert the scaled features to a dataframe and check the head of this dataframe to make sure the scaling worked.

# In[44]:


df_feat = pd.DataFrame(scaled_features,columns=df.columns[:-1])
df_feat.head()


# ## Train Test Split
# Use train_test_split to split your data into a training set and a testing set.

# In[45]:


from sklearn.model_selection import train_test_split


# In[46]:


X_train, X_test, y_train, y_test = train_test_split(scaled_features,                 df['TARGET CLASS'],test_size=0.30)


# ## Using KNN
# Import KNeighborsClassifier from scikit learn.

# In[47]:


from sklearn.neighbors import KNeighborsClassifier


# In[48]:


#Create a KNN model instance with n_neighbors=1
knn = KNeighborsClassifier(n_neighbors=1)


# In[49]:


# Fit this KNN model to the training data.
knn.fit(X_train,y_train)


# ## Predictions and Evaluations
# Let's evaluate our KNN model!
# 
# Use the predict method to predict values using your KNN model and X_test.

# In[50]:


pred = knn.predict(X_test)


# In[51]:


#** Create a confusion matrix and classification report.**
from sklearn.metrics import classification_report,confusion_matrix
print(confusion_matrix(y_test,pred))


# In[52]:


print(classification_report(y_test,pred))


# ## Choosing a K Value
# Let's go ahead and use the elbow method to pick a good K Value!
# 
# ** Create a for loop that trains various KNN models with different k values, then keep track of the error_rate for each of these models with a list. Refer to the lecture if you are confused on this step.

# In[65]:


error_rate = []

# Will take some time
for i in range(1,40):
    
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
    #print(np.mean(pred_i != y_test))


# In[69]:


#Now create the following plot using the information from your for loop.
plt.figure(figsize=(10,6))
plt.plot(range(1,40),error_rate,color='blue', linestyle='dashed',         marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')


# ## Retrain with new K Value
# Retrain your model with the best K value (up to you to decide what you want) and re-do the classification report and the confusion matrix.

# In[72]:


# NOW WITH K=12
knn = KNeighborsClassifier(n_neighbors=6)

knn.fit(X_train,y_train)
pred = knn.predict(X_test)

print('WITH K=36')
print('\n')
print('Confusion_Matrix:')
print(confusion_matrix(y_test,pred))
print('\n')
print('Confusion_report:')
print(classification_report(y_test,pred))


# In[ ]:





# In[ ]:





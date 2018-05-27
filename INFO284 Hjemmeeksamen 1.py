
# coding: utf-8

# In[87]:


get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import scipy.stats as stats
from pandas import DataFrame as df

file_handler = open("Flaveria.csv", "r")

# parses the csv data into a pandas data frame
dataset = pd.read_csv(file_handler, header=0)

dataset.replace(["L", "M", "H", "brownii", "pringlei", "trinervia", "ramosissima", "robusta", "bidentis"],
                    [1, 2, 3, 1, 2, 3, 4, 5, 6], inplace=True)

file_handler.close()


# In[88]:


from sklearn.model_selection import train_test_split


# In[89]:


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 2]


# In[90]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


# In[91]:


regressor = LinearRegression()
regressor.fit(X_train, y_train)


# In[92]:


y_pred = regressor.predict(X_test)


# In[93]:


print(y_pred)


# In[94]:


print(y_test)


# In[95]:


regressor.score(X_test, y_test, sample_weight=None)


# In[96]:


regressor.score(X_train, y_train)


# In[97]:


regressor.score(X_test, y_pred)


# In[98]:


regressor.score(X_test, y_test)


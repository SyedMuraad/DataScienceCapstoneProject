#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df = pd.read_csv('CAR DETAILS.csv')
df.head()


# In[3]:


df['fuel'].value_counts()


# In[4]:


df.columns


# In[5]:


df['km_driven'].value_counts()


# In[6]:


df['seller_type'].value_counts()


# In[7]:


df['name'].value_counts()


# In[8]:


df['selling_price'].value_counts()


# In[8]:


df['owner'].value_counts()


# In[9]:


df['transmission'].value_counts()


# In[11]:


df['year'].value_counts()


# In[10]:


df.isnull().sum()


# In[13]:


df.duplicated().sum()


# In[11]:


plt.scatter(x=df['fuel'],y=df['km_driven'])
plt.title('FUEL ENGINE VS KM DRIVEN',color='blue')
plt.show()


# ## Conclusion
# 1) Diesel and Petrol engine equipped vehicles have covered much distance as compared to the other variants(CNG,LPG and Electric)

# In[12]:


sns.countplot(x=df['seller_type'])
plt.show()


# ## Conclusion
# 1) Individual sellers are bulk in number than dealer type sellers and Trust Mark dealers.

# In[13]:


sns.displot(x=df['transmission'],color='red')
plt.show()


# ## Conclusion
# 1) Manual transmission loaded vehicles outnumbers the Automatic transmission ones.

# In[14]:


sns.boxplot(x=df['year'],y=df['selling_price'])
plt.xticks(rotation=90)
plt.show()


# ## Conclusion
# 1)There is an increasing trend over the years with respect to selling price owing to the new and innovative features

# In[15]:


df.nunique()


# In[16]:


df.head()


# In[21]:


df.head()


# In[28]:


df.drop(['year'],axis=1,inplace=True)


# In[29]:


df.head()


# # One Hot Encoding

# In[30]:


df = pd.get_dummies(df,drop_first=True)


# In[32]:


df.head()


# In[33]:


df.corr()


# In[34]:


x = df.iloc[:,1:]
y = df.iloc[:,0]


# In[35]:


x.head()


# In[36]:


y.head()


# In[40]:


from sklearn.ensemble import ExtraTreesRegressor
model=ExtraTreesRegressor()
model.fit(x,y)


# In[41]:


print(model.feature_importances_)


# In[43]:


feat_importances = pd.Series(model.feature_importances_,index=x.columns)
feat_importances.nlargest(5).plot(kind='barh')
plt.show()


# # Conclusion
# Transmission manual and fuel diesel are the most significant features

# # Train Test Splitting

# In[44]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2)


# In[48]:


x_train.shape


# In[51]:


from sklearn.ensemble import RandomForestRegressor
rf_random=RandomForestRegressor()


# In[52]:


import numpy as np
n_estimators=[int(x) for x in np.linspace(start=100,stop=1200,num=12)]
print(n_estimators)


# In[63]:


max_features = ['auto','sqrt']
max_depth = [int(x) for x in np.linspace(5,30, num = 6)]
min_samples_split = [2,5,10,15,100]
min_samples_leaf = [1,2,5,10]


# In[64]:


from sklearn.model_selection import RandomizedSearchCV


# In[65]:


random_grid = {'n_estimators':n_estimators,
              'max_features': max_features,
              'max_depth': max_depth,
              'min_samples_split':min_samples_split,
              'min_samples_leaf':min_samples_leaf}
print(random_grid)


# In[66]:


rf = RandomForestRegressor()


# In[67]:


rf_random = RandomizedSearchCV(estimator = rf,param_distributions = random_grid,scoring='neg_mean_squared_error',n_iter=10,cv = 5, verbose=2,random_state=42,n_jobs=1)
                              


# In[68]:


rf_random.fit(x_train,y_train)


# In[69]:


predictions=rf_random.predict(x_test)


# In[70]:


predictions


# In[71]:


sns.distplot(y_test-predictions)


# In[72]:


plt.scatter(y_test,predictions)


# In[73]:


import pickle
file = open('random_forest_regression_model.pkl','wb')
pickle.dump(rf_random,file)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd


# In[4]:


mb = pd.read_csv('Melbourne_housing_FULL.csv.zip')


# In[116]:


mb.head()


# In[115]:


mb.info()


# In[117]:


mb['Date'] = pd.to_datetime(mb['Date'])


# In[27]:


mb1 = mb.dropna(subset=['Price'])


# In[32]:


mb2 = mb1.drop(columns=['Postcode','Regionname','Lattitude','Longtitude'])


# In[33]:


mb2.head()


# In[34]:


mb3= mb2.drop(columns=['Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt'])


# In[35]:


mb3.head()


# In[36]:


mb3.describe()


# In[37]:


mb3.plot(x='Rooms', y='Price', kind='scatter')


# In[38]:


mb3.plot(x='Distance', y='Price', kind='scatter')


# In[39]:


mb3.plot(x='Propertycount', y='Price', kind='scatter')


# In[118]:


mb3 = mb3.fillna(value=0.00001)


# In[40]:


mb3.select_dtypes(['float64','int64']).columns


# In[41]:


mb3.info()


# In[42]:


X=mb3[['Rooms','Distance','Propertycount']]


# In[43]:


y=mb3['Price']


# In[44]:


from sklearn.model_selection import train_test_split


# In[45]:


X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.20, random_state=0)


# In[46]:


from sklearn.linear_model import LinearRegression


# In[47]:


model = LinearRegression()


# In[48]:


model.fit(X_train,y_train)


# In[49]:


y_predict=model.predict(X_test)


# In[50]:


from sklearn.metrics import mean_squared_error


# In[51]:


mean_squared_error(y_test,y_predict)


# In[53]:


from sklearn.model_selection import cross_val_score


# In[54]:


scores = cross_val_score(model, X, y, cv=10)


# In[55]:


scores


# In[56]:


scores.mean()


# In[57]:


from sklearn.decomposition import PCA


# In[58]:


pca=PCA()


# In[61]:


mb4=pd.get_dummies(mb, columns=['Type','Method'])


# In[65]:


mb4['Date'] = pd.to_datetime(mb4['Date'])


# In[73]:


mb5=mb4.drop(columns=['Regionname','CouncilArea','Date','SellerG','Address','Suburb'])


# In[75]:


mb5.select_dtypes(['float64','int64']).columns


# In[88]:


mb5= mb5.drop(columns=['Bedroom2','Bathroom','Car','Landsize','BuildingArea','YearBuilt'])


# In[89]:


mb5 = mb5.fillna(value=0.00001)


# In[90]:


mb5.head()


# In[107]:


mb6=pca.fit_transform(mb5)


# In[108]:


mb6


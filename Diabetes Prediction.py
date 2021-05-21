#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[26]:


data = pd.read_csv('F:/Python/Diabetes Prediction/Prediction.csv')


# In[27]:


data


# In[11]:


data.head(10)


# In[12]:


data.tail(10)


# In[13]:


data.isnull().sum()


# In[14]:


import seaborn as sns


# In[15]:


corrmat = data.corr()


# In[16]:


corrmat


# In[17]:


top_corr_features = corrmat.index


# In[18]:


top_corr_features


# In[23]:


plt.figure(figsize = (20,20))
sns.heatmap(data[top_corr_features].corr(), annot=True, cmap ="RdYlGn")


# # Changing the diabeties column data from boolean to number

# In[24]:


diabetes_map = {True: 1, False: 0}
data['diabetes'] = data['diabetes'].map(diabetes_map)


# In[25]:


data.head(10)


# In[28]:


diabetes_true_count = len(data.loc[data['diabetes'] == True])
diabetes_false_count = len(data.loc[data['diabetes'] == False])


# In[29]:


diabetes_true_count, diabetes_false_count


# In[30]:


data.columns


# In[31]:


#train test split

from sklearn.model_selection import train_test_split

feature_columns = ['num_preg', 'glucose_conc', 'diastolic_bp', 'thickness', 'insulin', 'bmi', 'diab_pred', 'age', 'skin']
predicted_class = ['diabetes']


# In[32]:


x = data[feature_columns].values
y = data[predicted_class].values

x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=.30, random_state=10)


# In[33]:


print("total number of rows : {0}".format(len(data)))
print("number of rows missing glucose_conc: {0}".format(len(data.loc[data['glucose_conc'] == 0])))
print("number of rows missing diastolic_bp: {0}".format(len(data.loc[data['diastolic_bp'] == 0])))
print("number of rows missing thickness: {0}".format(len(data.loc[data['thickness'] == 0])))
print("number of rows missing insulin: {0}".format(len(data.loc[data['insulin'] == 0])))
print("number of rows missing bmi: {0}".format(len(data.loc[data['bmi'] == 0])))
print("number of rows missing diab_pred: {0}".format(len(data.loc[data['diab_pred'] == 0])))
print("number of rows missing age: {0}".format(len(data.loc[data['age'] == 0])))
print("number of rows missing skin: {0}".format(len(data.loc[data['skin'] == 0])))


# In[37]:


from sklearn.impute import SimpleImputer

fill_values = SimpleImputer(missing_values=0, strategy="mean")

x_train = fill_values.fit_transform(x_train)
x_test = fill_values.fit_transform(x_test)


# In[38]:


x_train


# In[39]:


x_test


# In[40]:


from sklearn.ensemble import RandomForestClassifier

random_forest_model = RandomForestClassifier(random_state= 10)

random_forest_model.fit(x_train, y_train.ravel())


# In[41]:


predict_train_data = random_forest_model.predict(x_test)


# In[43]:


from sklearn import metrics

print("Acuracy = {0:3f}".format(metrics.accuracy_score(y_test,predict_train_data)))


# In[ ]:





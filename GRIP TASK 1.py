#!/usr/bin/env python
# coding: utf-8

# # **SPARKS FOUNDATION** #

# # **Task1** #
# **BATCH- #GRIPJULY2022**

# **Student's Performance Based On Study Hours**

# # **SURYAKANTA SUNDARAY** #
# **DATA SCIENCE AND BUSINESS ANALYST INTERN**

# **Dataset link and Description**
# http://bit.ly/w-data

# In[1]:


import pandas as pd
import numpy as np  
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# **Data Source**

# In[2]:


url = "http://bit.ly/w-data"
data = pd.read_csv(url)
print("Data imported successfully")

data.head(10)


# *Statistical status of data*

# In[3]:


data.describe()


# ## **Plotting the distribution of scores** ##

# In[4]:


data.plot(x='Hours', y='Scores', style='o')  
plt.title('Hours vs Percentage')  
plt.xlabel('Hours Studied')  
plt.ylabel('Percentage Score')  
plt.show()


# In[17]:


x = data.iloc[:, :-1].values  
y = data.iloc[:, 1].values


# In[18]:


from sklearn.model_selection import train_test_split  
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)


# In[19]:


from sklearn.linear_model import LinearRegression  
regressor = LinearRegression()  
regressor.fit (x_train, y_train) 

print("Training complete.")


# # **Plotting the regression line** #

# In[20]:


line = regressor.coef_*x+regressor.intercept_
plt.scatter(x, y)
plt.plot(x, line);
plt.show()


# **making the predictions**

# In[21]:


print(x_test) # Testing data - In Hours
y_pred = regressor.predict(x_test) # Predicting the scores


# **Comparing Actual vs Predicted**

# In[22]:


df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})  
df


# In[23]:


df.describe()


# # **predicting for the test dataset** #

# In[24]:


pred = regressor.predict(x_test.reshape(-1,1))
pred


# In[25]:


pred1 = pd.DataFrame(pred)
pred1


# In[26]:


plt.plot(x_test,pred,label='LinearRegression',color = 'b')
plt.scatter(x_test,y_test,label = 'Test_data',color = 'r')
plt.legend()
plt.show()


# In[27]:


regressor.predict([[9.25]])


# # **EVALUATING THE MODEL** #

# In[28]:


from sklearn import metrics  
print('Mean Absolute Error:', metrics.mean_absolute_error(y_test, y_pred))


# In[ ]:




